#include "application.h"
#include "cuda/cuda_utils.h"
#include <fstream>
#include <cstdio>
#include <ctime>

static Application* g_app = nullptr;

Application::Application()
    : m_window(nullptr), m_lastTime(0), m_dt(0)
    , m_frameCount(0), m_fpsTimer(0), m_fps(0), m_initialized(false) {
    g_app = this;
}

Application::~Application() { shutdown(); }

void Application::loadConfig() {
    // SANE DEFAULTS - Forces Barnes-Hut and disables Brute Force completely
    m_cfg = {
        50000,          // particleCount
        0.5f,           // G
        0.5f,           // softening
        0.005f,         // timestep
        0.8f,           // theta
        0.1f,           // mergeDistance
        false,          // bruteForce <--- THIS MUST BE FALSE
        true,           // evolution
        true,           // adaptiveTimestep
        true,           // adaptiveTheta
        false,          // volumetricEnabled
        64,             // densityFieldRes
        1600,           // winW
        900,            // winH
        "galaxy_collision", // scenario
        true,           // bloom <--- ENABLED
        0.8f,           // bloomThresh
        0.4f,           // bloomIntensity
        1.2f,           // exposure
        true,           // trails <--- ENABLED
        16,             // trailLen
        5000,           // maxTrailP
        0.2f,           // vignette
        0.002f,         // chromatic
        30.0f,          // camSpeed
        0.1f,           // camSens
        60.0f,          // camFOV
        0.01f,          // nearP
        50000.0f        // farP
    };

    std::ifstream file("config/simulation.json");
    if (file.is_open()) {
        std::string c((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        auto eF=[&](const std::string& k,float& v){size_t p=c.find("\""+k+"\"");if(p!=std::string::npos){p=c.find(":",p);if(p!=std::string::npos)v=std::stof(c.substr(p+1));}};
        auto eI=[&](const std::string& k,int& v){size_t p=c.find("\""+k+"\"");if(p!=std::string::npos){p=c.find(":",p);if(p!=std::string::npos)v=std::stoi(c.substr(p+1));}};
        auto eU=[&](const std::string& k,uint32_t& v){size_t p=c.find("\""+k+"\"");if(p!=std::string::npos){p=c.find(":",p);if(p!=std::string::npos)v=(uint32_t)std::stoul(c.substr(p+1));}};
        auto eS=[&](const std::string& k,std::string& v){size_t p=c.find("\""+k+"\"");if(p!=std::string::npos){p=c.find("\"",p+k.length()+2);if(p!=std::string::npos){size_t s=p+1,e=c.find("\"",s);if(e!=std::string::npos)v=c.substr(s,e-s);}}};
        auto eB=[&](const std::string& k,bool& v){size_t p=c.find("\""+k+"\"");if(p!=std::string::npos){p=c.find(":",p);if(p!=std::string::npos){std::string r=c.substr(p+1);size_t i=0;while(i<r.size()&&(r[i]==' '||r[i]=='\t'))i++;v=(r.substr(i,4)=="true");}}};

        eU("particle_count",m_cfg.particleCount);
        eF("gravitational_constant",m_cfg.G); eF("softening_length",m_cfg.softening);
        eF("timestep",m_cfg.timestep); eF("theta",m_cfg.theta); eF("merge_distance",m_cfg.mergeDistance);
        eB("use_brute_force",m_cfg.bruteForce); eB("stellar_evolution",m_cfg.evolution);
        eB("adaptive_timestep",m_cfg.adaptiveTimestep); eB("adaptive_theta",m_cfg.adaptiveTheta);
        eB("volumetric_enabled",m_cfg.volumetricEnabled); eI("density_field_res",m_cfg.densityFieldRes);
        eI("window_width",m_cfg.winW); eI("window_height",m_cfg.winH);
        eS("scenario",m_cfg.scenario);
        eB("bloom_enabled",m_cfg.bloom); eF("bloom_threshold",m_cfg.bloomThresh);
        eF("bloom_intensity",m_cfg.bloomIntensity); eF("exposure",m_cfg.exposure);
        eB("trails_enabled",m_cfg.trails); eU("trail_length",m_cfg.trailLen); eU("max_trail_particles",m_cfg.maxTrailP);
        eF("vignette_strength",m_cfg.vignette); eF("chromatic_strength",m_cfg.chromatic);
        eF("camera_speed",m_cfg.camSpeed); eF("camera_sensitivity",m_cfg.camSens);
        eF("camera_fov",m_cfg.camFOV); eF("near_plane",m_cfg.nearP); eF("far_plane",m_cfg.farP);
    }
    
    // Hard override to ensure performance
    m_cfg.bruteForce = false; 
    if (m_cfg.particleCount > 100000) m_cfg.particleCount = 100000;
}

bool Application::initWindow() {
    if (!glfwInit()) return false;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    m_window = glfwCreateWindow(m_cfg.winW, m_cfg.winH, "Universe Simulation", nullptr, nullptr);
    if (!m_window) { glfwTerminate(); return false; }
    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(0); // Uncap FPS
    glfwSetFramebufferSizeCallback(m_window, fbCallback);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return false;
    printf("GL: %s | %s\n", glGetString(GL_VERSION), glGetString(GL_RENDERER));
    return true;
}

bool Application::initCUDA() {
    int dc = 0; CUDA_CHECK(cudaGetDeviceCount(&dc)); if (!dc) return false;
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("CUDA: %s (SM %d.%d, %zuMB)\n", p.name, p.major, p.minor, p.totalGlobalMem/(1024*1024));
    CUDA_CHECK(cudaSetDevice(0));
    return true;
}

void Application::resetSimulation(Scenario sc) {
    m_renderer.shutdown();
    m_sim.cleanup();
    m_particles.deallocate();

    InitialConditions::generate(m_particles, sc, m_cfg.particleCount);

    RenderConfig rc;
    rc.width = m_cfg.winW; rc.height = m_cfg.winH;
    rc.bloomEnabled = m_cfg.bloom; 
    rc.bloomThreshold = m_cfg.bloomThresh;
    rc.bloomIntensity = m_cfg.bloomIntensity; 
    rc.exposure = m_cfg.exposure;
    rc.trailsEnabled = m_cfg.trails; 
    rc.trailLength = m_cfg.trailLen;
    rc.maxTrailParticles = m_cfg.maxTrailP;
    rc.vignetteStrength = m_cfg.vignette; 
    rc.chromaticStrength = m_cfg.chromatic;
    rc.volumetricEnabled = m_cfg.volumetricEnabled;
    rc.densityFieldRes = m_cfg.densityFieldRes;
    
    m_renderer.init(rc);
    m_renderer.setupInterop(m_particles);

    SimulationConfig sc2;
    sc2.G = m_cfg.G; sc2.softening = m_cfg.softening; sc2.timestep = m_cfg.timestep;
    sc2.theta = m_cfg.theta; sc2.mergeDistance = m_cfg.mergeDistance;
    sc2.bruteForce = m_cfg.bruteForce; sc2.evolution = m_cfg.evolution;
    sc2.adaptiveTimestep = m_cfg.adaptiveTimestep;
    sc2.adaptiveTheta = m_cfg.adaptiveTheta;
    sc2.volumetricEnabled = m_cfg.volumetricEnabled;
    sc2.densityFieldRes = m_cfg.densityFieldRes;
    
    m_sim.init(sc2, m_cfg.particleCount);
    m_sim.setParticles(&m_particles);
}

bool Application::init() {
    loadConfig();
    if (!initWindow() || !initCUDA()) return false;

    Scenario sc = scenarioFromString(m_cfg.scenario);
    InitialConditions::generate(m_particles, sc, m_cfg.particleCount);

    RenderConfig rc;
    rc.width = m_cfg.winW; rc.height = m_cfg.winH;
    rc.bloomEnabled = m_cfg.bloom; 
    rc.bloomThreshold = m_cfg.bloomThresh;
    rc.bloomIntensity = m_cfg.bloomIntensity; 
    rc.exposure = m_cfg.exposure;
    rc.trailsEnabled = m_cfg.trails; 
    rc.trailLength = m_cfg.trailLen;
    rc.maxTrailParticles = m_cfg.maxTrailP;
    rc.vignetteStrength = m_cfg.vignette; 
    rc.chromaticStrength = m_cfg.chromatic;
    rc.volumetricEnabled = m_cfg.volumetricEnabled;
    rc.densityFieldRes = m_cfg.densityFieldRes;
    
    m_renderer.init(rc);
    m_renderer.setupInterop(m_particles);

    SimulationConfig sc2;
    sc2.G = m_cfg.G; sc2.softening = m_cfg.softening; sc2.timestep = m_cfg.timestep;
    sc2.theta = m_cfg.theta; sc2.mergeDistance = m_cfg.mergeDistance;
    sc2.bruteForce = false; // MUST BE FALSE FOR PERFORMANCE
    sc2.evolution = m_cfg.evolution;
    sc2.adaptiveTimestep = m_cfg.adaptiveTimestep;
    sc2.adaptiveTheta = m_cfg.adaptiveTheta;
    sc2.volumetricEnabled = m_cfg.volumetricEnabled;
    sc2.densityFieldRes = m_cfg.densityFieldRes;
    
    m_sim.init(sc2, m_cfg.particleCount);
    m_sim.setParticles(&m_particles);

    m_camera.setPosition(glm::vec3(0.0f, 0.0f, 60.0f)); 
    m_camera.setTarget(glm::vec3(0.0f));
    m_camera.setFOV(60.0f);
    m_camera.setAspectRatio((float)m_cfg.winW / (float)m_cfg.winH);
    m_camera.setNearFar(0.01f, 50000.0f);
    m_camera.setSpeed(30.0f);
    m_camera.setSensitivity(0.1f);

    m_sim.resume();
    m_input.init(m_window);
    m_initialized = true;
    return true;
}

void Application::run() {
    m_lastTime = (float)glfwGetTime();
    mainLoop();
}

void Application::mainLoop() {
    while (!glfwWindowShouldClose(m_window) && !m_input.shouldClose()) {
        float now = (float)glfwGetTime();
        m_dt = now - m_lastTime;
        m_lastTime = now;
        m_dt = fminf(m_dt, 0.1f);

        m_fpsTimer += m_dt; m_frameCount++;
        if (m_fpsTimer >= 1.0f) {
            m_fps = m_frameCount / m_fpsTimer;
            m_frameCount = 0; m_fpsTimer = 0;
        }

        glfwPollEvents();
        m_input.processInput(m_window, m_camera, m_dt);

        if (m_input.isPauseToggled()) { if (m_sim.isPaused()) m_sim.resume(); else m_sim.pause(); }
        
        // --- LIVE TOGGLES ---
        if (m_input.isBloomToggled()) {
            m_cfg.bloom = !m_cfg.bloom;
            m_renderer.config().bloomEnabled = m_cfg.bloom;
        }
        if (m_input.isTrailsToggled()) {
            m_cfg.trails = !m_cfg.trails;
            m_renderer.config().trailsEnabled = m_cfg.trails;
        }
        if (m_input.isEvolutionToggled()) {
            m_cfg.evolution = !m_cfg.evolution;
            m_sim.config().evolution = m_cfg.evolution;
        }
        if (m_input.isVolumetricToggled()) {
            m_cfg.volumetricEnabled = !m_cfg.volumetricEnabled;
            m_sim.config().volumetricEnabled = m_cfg.volumetricEnabled;
            m_renderer.config().volumetricEnabled = m_cfg.volumetricEnabled;
        }
        if (m_input.isOverlayToggled()) {
            m_renderer.setOverlayVisible(!m_renderer.isOverlayVisible());
        }

        int ss = m_input.getScenarioSwitch();
        if (ss >= 0 && ss <= 3) {
            Scenario scenarios[] = {Scenario::BIG_BANG, Scenario::GALAXY_COLLISION, Scenario::PROTOGALACTIC_CLOUD, Scenario::SOLAR_SYSTEM};
            resetSimulation(scenarios[ss]);
        }

        m_camera.update(m_dt);
        m_sim.step();

        m_renderer.updateRenderBuffer(m_particles);
        m_renderer.findBlackHoles(m_particles, m_camera);

        m_renderer.beginFrame();
        m_renderer.renderParticles(m_camera, m_particles.count());
        m_renderer.endFrame(m_window, m_camera);

        const char* modes[] = {"FREE","ORBIT","OVERVIEW","CINEMATIC"};
        OverlayStats st;
        st.fps = m_fps; st.particleCount = m_particles.count();
        st.simTime = m_sim.getTime(); st.simStep = m_sim.getStep();
        st.treeBuildMs = m_sim.treeBuildMs(); st.forceCalcMs = m_sim.forceCalcMs();
        st.integrationMs = m_sim.integrationMs(); st.totalStepMs = m_sim.totalStepMs();
        st.frameTimeMs = m_dt * 1000.0f; st.paused = m_sim.isPaused();
        st.recording = false; st.recordedFrames = 0;
        st.cameraMode = modes[(int)m_camera.getMode()]; st.scenario = m_cfg.scenario;
        st.bloomOn = m_cfg.bloom; 
        st.trailsOn = m_cfg.trails; 
        st.evolutionOn = m_cfg.evolution; 
        st.volumetricOn = m_cfg.volumetricEnabled;
        st.diag = m_sim.diagnostics();
        m_renderer.renderOverlay(st);

        glfwSwapBuffers(m_window);
    }
}

void Application::fbCallback(GLFWwindow* w, int width, int height) {
    if (!width || !height || !g_app) return;
    g_app->m_renderer.resize(width, height);
    g_app->m_camera.setAspectRatio((float)width/height);
}

void Application::shutdown() {
    if (!m_initialized) return;
    m_renderer.shutdown();
    m_sim.cleanup();
    m_particles.deallocate();
    if (m_window) { glfwDestroyWindow(m_window); m_window = nullptr; }
    glfwTerminate();
    m_initialized = false;
}