#include "mesh/quad_mesh.h"
#include "common/common.h"
#include "common/file_helper.h"

QuadMesh::QuadMesh() : vertices_(Matrix2Xr::Zero(2, 0)), faces_(Matrix4Xi::Zero(4, 0)) {}

static const std::vector<std::string> ParseNumberFromLine(const std::string& line) {
    const std::string line_ext = " " + line + " ";
    int word_begin = -1;
    const int len = static_cast<int>(line_ext.size());
    std::vector<std::string> ret;
    for (int i = 0; i < len; ++i) {
        const char ch = line_ext[i];
        if (('0' <= ch && ch <= '9') || (ch == '.')) {
            // Part of the number.
            if (word_begin == -1) word_begin = i;
        } else {
            // Skip.
            if (word_begin != -1) {
                ret.push_back(line_ext.substr(word_begin, i - word_begin));
                word_begin = -1;
            }
        }
    }
    return ret;
}

void QuadMesh::Initialize(const Matrix2Xr& vertices, const Matrix4Xi& faces) {
    vertices_ = vertices;
    faces_ = faces;
}

void QuadMesh::LoadFromFile(const std::string& obj_file_name) {
    std::ifstream fin(obj_file_name);

    // Load all vertices.
    std::vector<std::array<real, 2>> vertices;
    std::vector<std::array<int, 4>> faces;
    std::string line;
    while (getline(fin, line)) {
        // Skip comments.
        if (line.empty() || line[0] == '#') continue;
        // Determine whether it is vertex or face.
        if (line[0] == 'v') {
            const std::vector<std::string> v_str = ParseNumberFromLine(line);
            CheckError(static_cast<int>(v_str.size()) == 3, "Found more numbers in " + line);
            std::array<real, 2> v;
            v[0] = ToReal(std::stod(v_str[0]));
            v[1] = ToReal(std::stod(v_str[1]));
            vertices.push_back(v);
        } else if (line[0] == 'f') {
            const std::vector<std::string> f_str = ParseNumberFromLine(line);
            CheckError(static_cast<int>(f_str.size()) == 4, "Found more numbers in " + line);
            std::array<int, 4> f;
            for (int i = 0; i < 4; ++i) f[i] = std::stoi(f_str[i]) - 1;
            faces.push_back(f);
        } else {
            PrintError("Unidentified line in " + obj_file_name + ": " + line);
        }
    }

    const int vertex_num = static_cast<int>(vertices.size());
    vertices_ = Matrix2Xr::Zero(2, vertex_num);
    for (int i = 0; i < vertex_num; ++i) {
        vertices_(0, i) = vertices[i][0];
        vertices_(1, i) = vertices[i][1];
    }
    const int face_num = static_cast<int>(faces.size());
    faces_ = Matrix4Xi::Zero(4, face_num);
    for (int i = 0; i < face_num; ++i) {
        faces_(0, i) = faces[i][0];
        faces_(1, i) = faces[i][1];
        faces_(2, i) = faces[i][2];
        faces_(3, i) = faces[i][3];
    }
}

void QuadMesh::SaveToFile(const std::string& obj_file_name) const {
    PrepareToCreateFile(obj_file_name);
    std::ofstream fout(obj_file_name);

    // Write all vertices.
    const int vertex_num = NumOfVertices();
    for (int i = 0; i < vertex_num; ++i) {
        fout << "v " << vertices_(0, i) << " " << vertices_(1, i) << " 0" << std::endl;
    }
    // Write all faces.
    const int face_num = NumOfFaces();
    for (int i = 0; i < face_num; ++i) {
        fout << "f " << faces_(0, i) + 1 << " " << faces_(1, i) + 1
            << " " << faces_(2, i) + 1 << " " << faces_(3, i) + 1 << std::endl;
    }
}