#include "shader_manager.h"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>

ShaderManager::ShaderManager() {}

ShaderManager::~ShaderManager() {
    deleteAll();
}

GLuint ShaderManager::loadProgram(const std::string& name,
                                   const std::string& vertPath,
                                   const std::string& fragPath) {
    GLuint vert = compileShader(GL_VERTEX_SHADER, vertPath);
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragPath);
    GLuint prog = linkProgram(vert, frag);
    glDeleteShader(vert);
    glDeleteShader(frag);
    m_programs[name] = prog;
    return prog;
}

GLuint ShaderManager::getProgram(const std::string& name) const {
    auto it = m_programs.find(name);
    if (it != m_programs.end()) return it->second;
    return 0;
}

void ShaderManager::useProgram(const std::string& name) const {
    auto it = m_programs.find(name);
    if (it != m_programs.end()) {
        glUseProgram(it->second);
    }
}

void ShaderManager::deleteAll() {
    for (auto& p : m_programs) {
        glDeleteProgram(p.second);
    }
    m_programs.clear();
}

GLuint ShaderManager::compileShader(GLenum type, const std::string& path) {
    std::string source = readFile(path);
    const char* src = source.c_str();

    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, log);
        fprintf(stderr, "Shader compilation error (%s):\n%s\n", path.c_str(), log);
        exit(EXIT_FAILURE);
    }
    return shader;
}

GLuint ShaderManager::linkProgram(GLuint vert, GLuint frag) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);

    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[1024];
        glGetProgramInfoLog(program, 1024, nullptr, log);
        fprintf(stderr, "Shader link error:\n%s\n", log);
        exit(EXIT_FAILURE);
    }
    return program;
}

std::string ShaderManager::readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "Cannot open shader file: %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}