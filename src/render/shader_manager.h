#pragma once

#include <glad/glad.h>
#include <string>
#include <unordered_map>

class ShaderManager {
public:
    ShaderManager();
    ~ShaderManager();

    GLuint loadProgram(const std::string& name,
                       const std::string& vertPath,
                       const std::string& fragPath);
    GLuint getProgram(const std::string& name) const;
    void useProgram(const std::string& name) const;
    void deleteAll();

private:
    GLuint compileShader(GLenum type, const std::string& path);
    GLuint linkProgram(GLuint vert, GLuint frag);
    std::string readFile(const std::string& path);

    std::unordered_map<std::string, GLuint> m_programs;
};