#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "application.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>

#include <vk_mem_alloc.h>

namespace application {

    namespace {

        constexpr float PI = 3.14159265f;
        struct Vertex {
            glm::vec3 position;
            glm::vec3 normal;
            glm::vec3 color;
        };

        struct MeshData {
            std::vector<Vertex> vertices;
            std::vector<uint32_t> indices;
        };

      
        void generateFlatNormals(MeshData& mesh) {
            for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
                Vertex& a = mesh.vertices[mesh.indices[i + 0]];
                Vertex& b = mesh.vertices[mesh.indices[i + 1]];
                Vertex& c = mesh.vertices[mesh.indices[i + 2]];

                const glm::vec3 normal = glm::normalize(glm::cross(b.position - a.position,
                                                                   c.position - a.position));
                a.normal = normal;
                b.normal = normal;
                c.normal = normal;
            }
        }

 
        MeshData makeCube() {
            const glm::vec3 corners[] = {
                { -0.5f, -0.5f, -0.5f },
                {  0.5f, -0.5f, -0.5f },
                {  0.5f,  0.5f, -0.5f },
                { -0.5f,  0.5f, -0.5f },
                { -0.5f, -0.5f,  0.5f },
                {  0.5f, -0.5f,  0.5f },
                {  0.5f,  0.5f,  0.5f },
                { -0.5f,  0.5f,  0.5f },
            };

   
            const uint32_t cube_indices[] = {
                0, 3, 2,  2, 1, 0, // -z
                4, 5, 6,  6, 7, 4, // +z
                0, 4, 7,  7, 3, 0, // -x
                1, 2, 6,  6, 5, 1, // +x
                3, 7, 6,  6, 2, 3, // +y
                0, 1, 5,  5, 4, 0, // -y
            };


            MeshData mesh;
            for (uint32_t index : cube_indices) {
                mesh.indices.push_back(uint32_t(mesh.vertices.size()));
                mesh.vertices.push_back({ corners[index], glm::vec3(0.0f), glm::vec3(1.0f) });
            }

            generateFlatNormals(mesh);
            return mesh;
        }

        MeshData makeSphere(int stacks, int slices) {
            MeshData mesh;

            for (int i = 0; i <= stacks; ++i) {
                const float theta = PI * float(i) / float(stacks);
                for (int j = 0; j <= slices; ++j) {
                    const float phi = 2.0f * PI * float(j) / float(slices);
                    const glm::vec3 p = {
                        std::sin(theta) * std::cos(phi),
                        std::cos(theta),
                        std::sin(theta) * std::sin(phi),
                    };
                    mesh.vertices.push_back({ p, p, glm::vec3(1.0f) });
                }
            }

            const uint32_t row = uint32_t(slices + 1);
            for (int i = 0; i < stacks; ++i) {
                for (int j = 0; j < slices; ++j) {
                    const uint32_t a = uint32_t(i) * row + uint32_t(j);
                    const uint32_t b = a + row;
                    mesh.indices.insert(mesh.indices.end(), { a, a + 1, b, a + 1, b + 1, b });
                }
            }

            return mesh;
        }

  
        struct SceneUniforms {
            glm::mat4 view_projection;              // 0
            glm::vec3 camera_position;              // 64
            float _padding0;
            glm::vec3 directional_light_direction;  // 80
            float _padding1;
            glm::vec3 directional_light_color;      // 96
            float _padding2;
            glm::vec3 ambient_light;                // 112
            uint32_t point_light_count;             // 124
            uint32_t spot_light_count;              // 128
            uint32_t _padding3[3];
        };
        static_assert(offsetof(SceneUniforms, point_light_count) == 124, "std140 layout");
        static_assert(offsetof(SceneUniforms, spot_light_count) == 128, "std140 layout");

        struct ModelUniform {
            glm::mat4 model;                        // 0
            glm::vec3 albedo;                       // 64
            float _padding;
            glm::vec3 specular;                     // 80
            float alpha;                            // 92
        };
        static_assert(sizeof(ModelUniform) == 96, "std140 layout");

        struct PointLight {
            glm::vec3 position;
            float intensity;
            glm::vec3 color;
            float _padding;
        };
        static_assert(sizeof(PointLight) == 32, "std430 layout");

        struct SpotLight {
            glm::vec3 position;
            float intensity;
            glm::vec3 color;
            float inner_angle;  
            glm::vec3 direction;
            float outer_angle;  
        };
        static_assert(sizeof(SpotLight) == 48, "std430 layout");

        struct LightPushConstants {
            glm::mat4 model;
            glm::vec4 color;
        };

        struct Buffer {
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            void* mapped = nullptr;
            VkDeviceSize size = 0;
        };

        struct Mesh {
            Buffer vertex_buffer;
            Buffer index_buffer;
            uint32_t index_count = 0;
        };

        bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, const void* data, Buffer& out) {
            const VkBufferCreateInfo buffer_info = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = size,
                .usage = usage,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };

            const VmaAllocationCreateInfo alloc_info = {
                .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                       | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO,
            };

            VmaAllocationInfo allocation_info{};

            if (vmaCreateBuffer(graphics::internal::context.allocator,
                &buffer_info, &alloc_info,
                &out.buffer, &out.allocation,
                &allocation_info) != VK_SUCCESS) {
                std::cerr << "Failed to create buffer\n";
                return false;
            }

            out.mapped = allocation_info.pMappedData;
            out.size = size;

            if (data != nullptr) {
                std::memcpy(out.mapped, data, size_t(size));
            }
            return true;
        }

        void destroyBuffer(Buffer& buffer) {
            if (buffer.buffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(graphics::internal::context.allocator, buffer.buffer, buffer.allocation);
            }
            buffer = {};
        }

        
        bool ensureBufferSize(Buffer& buffer, VkDeviceSize needed, VkBufferUsageFlags usage) {
            if (buffer.buffer != VK_NULL_HANDLE && buffer.size >= needed) {
                return false;
            }

            VkDeviceSize new_size = 256;
            while (new_size < needed) {
                new_size *= 2;
            }

            destroyBuffer(buffer);
            createBuffer(new_size, usage, nullptr, buffer);
            return true;
        }

        bool createMesh(const MeshData& data, Mesh& mesh) {
            if (!createBuffer(data.vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                data.vertices.data(), mesh.vertex_buffer)) {
                return false;
            }
            if (!createBuffer(data.indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                data.indices.data(), mesh.index_buffer)) {
                return false;
            }
            mesh.index_count = uint32_t(data.indices.size());
            return true;
        }

        void destroyMesh(Mesh& mesh) {
            destroyBuffer(mesh.vertex_buffer);
            destroyBuffer(mesh.index_buffer);
            mesh.index_count = 0;
        }

    
        struct Camera {
            glm::vec3 position = glm::vec3(4.0f, 3.0f, 7.0f);
            float yaw = -30.0f;          
            float pitch = -20.0f;        
            float fov = 45.0f;           
            float near_plane = 0.1f;
            float far_plane = 100.0f;
            int projection_mode = 0;   
            float ortho_size = 3.0f;
      
        };

        struct DirectionalLightSettings {
            bool enabled = true;
            float yaw = 35.0f;      
            float pitch = -50.0f;
            glm::vec3 color = glm::vec3(1.0f, 0.95f, 0.85f);
            float intensity = 0.25f;
        };

        struct AmbientLightSettings {
            glm::vec3 color = glm::vec3(0.55f, 0.65f, 1.0f);
            float intensity = 0.08f;
        };

        struct PointLightSettings {
            bool enabled = true;
            glm::vec3 position = glm::vec3(0.0f, 1.5f, 1.5f);
            glm::vec3 color = glm::vec3(1.0f);
            float intensity = 3.0f;
            bool orbit = false;         
            float orbit_speed = 0.7f;    
        };

        struct SpotLightSettings {
            bool enabled = true;
            glm::vec3 position = glm::vec3(0.0f, 3.0f, 0.0f);
            float yaw = 0.0f;            
            float pitch = -89.0f;
            glm::vec3 color = glm::vec3(1.0f);
            float intensity = 15.0f;
            float inner_angle = 15.0f;   
            float outer_angle = 25.0f;   
            bool attach_to_camera = false;
        };

        struct SceneObject {
            const char* name;
            Mesh* mesh;
            glm::vec3 position;
            glm::vec3 rotation;           
            glm::vec3 scale;
            glm::vec3 albedo;
            glm::vec3 specular;
            float alpha;

            bool animate;
            float phase;

            Buffer uniform_buffer;
            VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
            ModelUniform uniforms;
        };

        Camera camera;
        DirectionalLightSettings directional_light;
        AmbientLightSettings ambient_light;
        std::vector<PointLightSettings> point_lights;
        std::vector<SpotLightSettings> spot_lights;
        std::vector<SceneObject> objects;


        bool is_playing = true;
        float anim_speed = 1.0f;
        float anim_radius = 0.5f;
        float anim_height = 0.3f;
        float anim_angle = 0.0f;


        bool show_lights = true;
        float light_sphere_size = 0.08f;
        float spot_cone_length = 1.0f;


        Mesh cube_mesh;
        Mesh sphere_mesh;

        Buffer scene_uniform_buffer;
        Buffer point_light_buffer;
        Buffer spot_light_buffer;
        Buffer line_vertex_buffer;

        VkDescriptorSetLayout vk_scene_descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout vk_model_descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet vk_scene_descriptor_set = VK_NULL_HANDLE;

        VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;       
        VkPipelineLayout vk_light_pipeline_layout = VK_NULL_HANDLE; 
        VkPipeline vk_pipeline = VK_NULL_HANDLE;
        VkPipeline vk_light_pipeline = VK_NULL_HANDLE;              
        VkPipeline vk_line_pipeline = VK_NULL_HANDLE;               

        SceneUniforms scene_uniforms{};
        std::vector<PointLight> point_light_data;
        std::vector<SpotLight> spot_light_data;
        std::vector<LightPushConstants> light_spheres;
        std::vector<Vertex> line_vertices;

      
        glm::vec3 directionFromAngles(float yaw_degrees, float pitch_degrees) {
            const float yaw = glm::radians(yaw_degrees);
            const float pitch = glm::radians(pitch_degrees);
            return {
                std::sin(yaw) * std::cos(pitch),
                std::sin(pitch),
                -std::cos(yaw) * std::cos(pitch),
            };
        }

        void lookAtPoint(const glm::vec3& target) {
            const glm::vec3 d = glm::normalize(target - camera.position);
            camera.yaw = glm::degrees(std::atan2(d.x, -d.z));
            camera.pitch = glm::degrees(std::asin(std::clamp(d.y, -1.0f, 1.0f)));
        }


        VkShaderModule loadShaderModule(const char* name) {
            const char* directories[] = {
                "shaders/",
                "../shaders/",
                "../../shaders/",
                "../../../shaders/",
            };

            std::ifstream file;
            for (const char* directory : directories) {
                file.open(std::string(directory) + name, std::ios::binary | std::ios::ate);
                if (file) {
                    break;
                }
                file.clear();
            }

            if (!file) {
                std::cerr << "Failed to open shader file: " << name << '\n';
                return VK_NULL_HANDLE;
            }

            const size_t size = static_cast<size_t>(file.tellg());
            std::vector<uint32_t> buffer(size / sizeof(uint32_t));

            file.seekg(0);
            file.read(reinterpret_cast<char*>(buffer.data()), std::streamsize(size));
            file.close();

            const VkShaderModuleCreateInfo info = {
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = size,
                .pCode = buffer.data(),
            };

            VkShaderModule result = VK_NULL_HANDLE;
            if (vkCreateShaderModule(graphics::internal::context.device,
                &info, nullptr, &result) != VK_SUCCESS) {
                std::cerr << "Failed to create shader module: " << name << '\n';
                return VK_NULL_HANDLE;
            }
            return result;
        }

        void writeBufferDescriptor(VkDescriptorSet set, uint32_t binding,
                                   VkDescriptorType type, VkBuffer buffer, VkDeviceSize range) {
            const VkDescriptorBufferInfo buffer_info = {
                .buffer = buffer,
                .offset = 0,
                .range = range,
            };

            const VkWriteDescriptorSet write = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = set,
                .dstBinding = binding,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = type,
                .pBufferInfo = &buffer_info,
            };

            vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
        }


        SceneObject makeObject(const char* name, Mesh* mesh, glm::vec3 position, float scale,
                               glm::vec3 albedo, glm::vec3 specular, float alpha, float phase) {
            SceneObject object{};
            object.name = name;
            object.mesh = mesh;
            object.position = position;
            object.rotation = glm::vec3(0.0f);
            object.scale = glm::vec3(scale);
            object.albedo = albedo;
            object.specular = specular;
            object.alpha = alpha;
            object.animate = true;
            object.phase = phase;
            return object;
        }

        void resetLights() {
            directional_light = {};
            ambient_light = {};

            point_lights.clear();
            {
                PointLightSettings red;
                red.position = glm::vec3(1.8f, 0.3f, 1.5f);
                red.color = glm::vec3(1.0f, 0.25f, 0.1f);
                red.intensity = 3.0f;
                red.orbit = true;
                point_lights.push_back(red);

                PointLightSettings blue;
                blue.position = glm::vec3(-1.8f, 1.2f, -1.5f);
                blue.color = glm::vec3(0.1f, 0.6f, 1.0f);
                blue.intensity = 4.0f;
                blue.orbit = true;
                blue.orbit_speed = -0.5f;
                point_lights.push_back(blue);
            }

            spot_lights.clear();
            {

                SpotLightSettings top;
                top.position = glm::vec3(0.0f, 3.2f, 1.6f);
                top.yaw = 0.0f;
                top.pitch = -66.0f;
                top.color = glm::vec3(1.0f, 0.9f, 0.7f);
                top.intensity = 14.0f;
                spot_lights.push_back(top);


                SpotLightSettings side;
                side.position = glm::vec3(4.0f, 2.5f, 0.0f);
                side.yaw = -90.0f;
                side.pitch = -30.0f;
                side.color = glm::vec3(0.8f, 0.3f, 1.0f);
                side.intensity = 15.0f;
                side.inner_angle = 10.0f;
                side.outer_angle = 30.0f;
                spot_lights.push_back(side);
            }
        }

\
        bool createDescriptors() {
            auto& context = graphics::internal::context;

            const VkDescriptorSetLayoutBinding scene_bindings[] = {
                {
                    .binding = 0,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                },
                {
                    .binding = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                },
                {
                    .binding = 2,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                },
            };

            const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 3,
                .pBindings = scene_bindings,
            };

            if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr,
                &vk_scene_descriptor_set_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create scene descriptor set layout\n";
                return false;
            }

            const VkDescriptorSetLayoutBinding model_binding = {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            };

            const VkDescriptorSetLayoutCreateInfo model_layout_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 1,
                .pBindings = &model_binding,
            };

            if (vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr,
                &vk_model_descriptor_set_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create model descriptor set layout\n";
                return false;
            }


            const uint32_t object_count = uint32_t(objects.size());

            const VkDescriptorPoolSize pool_sizes[] = {
                {
                    .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1 + object_count,
                },
                {
                    .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 2,
                },
            };

            const VkDescriptorPoolCreateInfo pool_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 1 + object_count,
                .poolSizeCount = 2,
                .pPoolSizes = pool_sizes,
            };

            if (vkCreateDescriptorPool(context.device, &pool_info, nullptr,
                &vk_descriptor_pool) != VK_SUCCESS) {
                std::cerr << "Failed to create descriptor pool\n";
                return false;
            }


            const VkDescriptorSetAllocateInfo scene_alloc_info = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = vk_descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &vk_scene_descriptor_set_layout,
            };

            if (vkAllocateDescriptorSets(context.device, &scene_alloc_info,
                &vk_scene_descriptor_set) != VK_SUCCESS) {
                std::cerr << "Failed to allocate scene descriptor set\n";
                return false;
            }

            writeBufferDescriptor(vk_scene_descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                scene_uniform_buffer.buffer, sizeof(SceneUniforms));
            writeBufferDescriptor(vk_scene_descriptor_set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                point_light_buffer.buffer, VK_WHOLE_SIZE);
            writeBufferDescriptor(vk_scene_descriptor_set, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                spot_light_buffer.buffer, VK_WHOLE_SIZE);


            for (SceneObject& object : objects) {
                const VkDescriptorSetAllocateInfo model_alloc_info = {
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    .descriptorPool = vk_descriptor_pool,
                    .descriptorSetCount = 1,
                    .pSetLayouts = &vk_model_descriptor_set_layout,
                };

                if (vkAllocateDescriptorSets(context.device, &model_alloc_info,
                    &object.descriptor_set) != VK_SUCCESS) {
                    std::cerr << "Failed to allocate model descriptor set\n";
                    return false;
                }

                writeBufferDescriptor(object.descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    object.uniform_buffer.buffer, sizeof(ModelUniform));
            }


            const VkDescriptorSetLayout set_layouts[] = {
                vk_scene_descriptor_set_layout,
                vk_model_descriptor_set_layout,
            };

            const VkPipelineLayoutCreateInfo pipeline_layout_info = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 2,
                .pSetLayouts = set_layouts,
            };

            if (vkCreatePipelineLayout(context.device, &pipeline_layout_info, nullptr,
                &vk_pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create pipeline layout\n";
                return false;
            }

            const VkPushConstantRange push_range = {
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                .offset = 0,
                .size = sizeof(LightPushConstants),
            };

            const VkPipelineLayoutCreateInfo light_layout_info = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 1,
                .pSetLayouts = &vk_scene_descriptor_set_layout,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &push_range,
            };

            if (vkCreatePipelineLayout(context.device, &light_layout_info, nullptr,
                &vk_light_pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create light pipeline layout\n";
                return false;
            }

            return true;
        }

        VkPipeline createPipeline(const char* vertex_shader, const char* fragment_shader,
                                  VkPipelineLayout layout, VkPrimitiveTopology topology,
                                  VkCullModeFlags cull_mode) {
            auto& context = graphics::internal::context;

            VkShaderModule vert_module = loadShaderModule(vertex_shader);
            VkShaderModule frag_module = loadShaderModule(fragment_shader);
            if (vert_module == VK_NULL_HANDLE || frag_module == VK_NULL_HANDLE) {
                return VK_NULL_HANDLE;
            }

            const VkPipelineShaderStageCreateInfo stages[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vert_module,
                    .pName = "main",
                },
                {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = frag_module,
                    .pName = "main",
                },
            };

            const VkVertexInputBindingDescription vertex_binding = {
                .binding = 0,
                .stride = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };


            const VkVertexInputAttributeDescription vertex_attributes[] = {
                {
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, position),
                },
                {
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, normal),
                },
                {
                    .location = 2,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, color),
                },
            };

            const VkPipelineVertexInputStateCreateInfo vertex_input = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &vertex_binding,
                .vertexAttributeDescriptionCount = 3,
                .pVertexAttributeDescriptions = vertex_attributes,
            };

            const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = topology,
                .primitiveRestartEnable = VK_FALSE,
            };

            const VkPipelineViewportStateCreateInfo viewport_state = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .viewportCount = 1,
                .scissorCount = 1,
            };

  
            const VkPipelineRasterizationStateCreateInfo rasterization = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .depthClampEnable = VK_FALSE,
                .rasterizerDiscardEnable = VK_FALSE,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = cull_mode,
                .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                .depthBiasEnable = VK_FALSE,
                .lineWidth = 1.0f,
            };

            const VkPipelineMultisampleStateCreateInfo multisample = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
                .sampleShadingEnable = VK_FALSE,
            };

            const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = VK_TRUE,
                .depthWriteEnable = VK_TRUE,
                .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
                .depthBoundsTestEnable = VK_FALSE,
                .stencilTestEnable = VK_FALSE,
            };

            const VkPipelineColorBlendAttachmentState color_attachment = {
                .blendEnable = VK_FALSE,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT
                                | VK_COLOR_COMPONENT_G_BIT
                                | VK_COLOR_COMPONENT_B_BIT
                                | VK_COLOR_COMPONENT_A_BIT,
            };

            const VkPipelineColorBlendStateCreateInfo color_blend = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .logicOpEnable = VK_FALSE,
                .attachmentCount = 1,
                .pAttachments = &color_attachment,
            };

            const VkDynamicState dynamic_states[] = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR,
            };

            const VkPipelineDynamicStateCreateInfo dynamic_state = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .dynamicStateCount = 2,
                .pDynamicStates = dynamic_states,
            };

            const VkGraphicsPipelineCreateInfo pipeline_info = {
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .stageCount = 2,
                .pStages = stages,
                .pVertexInputState = &vertex_input,
                .pInputAssemblyState = &input_assembly,
                .pViewportState = &viewport_state,
                .pRasterizationState = &rasterization,
                .pMultisampleState = &multisample,
                .pDepthStencilState = &depth_stencil,
                .pColorBlendState = &color_blend,
                .pDynamicState = &dynamic_state,
                .layout = layout,
                .renderPass = context.render_pass,
                .subpass = 0,
            };

            VkPipeline pipeline = VK_NULL_HANDLE;
            if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info,
                nullptr, &pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create graphics pipeline\n";
                pipeline = VK_NULL_HANDLE;
            }

            vkDestroyShaderModule(context.device, vert_module, nullptr);
            vkDestroyShaderModule(context.device, frag_module, nullptr);

            return pipeline;
        }


        void drawSceneWindow() {
            ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(340.0f, 480.0f), ImGuiCond_FirstUseEver);
            ImGui::Begin("Camera & Objects");


            if (ImGui::CollapsingHeader("1. Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::DragFloat3("Position", &camera.position.x, 0.05f);
                ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f);
                ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f);

                ImGui::RadioButton("Perspective", &camera.projection_mode, 0);
                ImGui::SameLine();
                ImGui::RadioButton("Orthographic", &camera.projection_mode, 1);

                if (camera.projection_mode == 0) {
                    ImGui::SliderFloat("FOV", &camera.fov, 10.0f, 120.0f);
                } else {
                    ImGui::SliderFloat("Ortho size", &camera.ortho_size, 0.5f, 20.0f);
                }
                ImGui::DragFloatRange2("Near / far", &camera.near_plane, &camera.far_plane,
                    0.05f, 0.01f, 500.0f);
                camera.near_plane = std::max(camera.near_plane, 0.01f);
                camera.far_plane = std::max(camera.far_plane, camera.near_plane + 0.1f);


                if (ImGui::Button("Look at center")) {
                    lookAtPoint(glm::vec3(0.0f));
                }
                ImGui::SameLine();
                if (ImGui::Button("Reset camera")) {
                    camera = {};
                }
            }


            if (ImGui::CollapsingHeader("2. Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::Button(is_playing ? "Pause" : "Play")) {
                    is_playing = !is_playing;
                }
                ImGui::SliderFloat("Speed", &anim_speed, -3.0f, 3.0f);
                ImGui::SliderFloat("Radius", &anim_radius, 0.0f, 3.0f);
                ImGui::SliderFloat("Height", &anim_height, -1.0f, 1.0f);
            }

            if (ImGui::CollapsingHeader("3. Objects & materials", ImGuiTreeNodeFlags_DefaultOpen)) {
                for (size_t i = 0; i < objects.size(); ++i) {
                    SceneObject& object = objects[i];

                    ImGui::PushID(int(i));
                    if (ImGui::TreeNode(object.name)) {
                        ImGui::SeparatorText("Transform");
                        ImGui::DragFloat3("Position", &object.position.x, 0.05f);
                        ImGui::SliderFloat3("Rotation", &object.rotation.x, -180.0f, 180.0f);
                        ImGui::DragFloat3("Scale", &object.scale.x, 0.02f, 0.05f, 20.0f);
                        ImGui::Checkbox("Move along trajectory", &object.animate);

                        ImGui::SeparatorText("Material (Blinn-Phong)");
                        ImGui::ColorEdit3("Albedo", &object.albedo.x);
                        ImGui::ColorEdit3("Specular", &object.specular.x);
                        ImGui::SliderFloat("Shininess (alpha)", &object.alpha, 1.0f, 512.0f, "%.1f",
                            ImGuiSliderFlags_Logarithmic);

                        ImGui::TreePop();
                    }
                    ImGui::PopID();
                }
            }

            ImGui::End();
        }

       void directionEditor(float* yaw, float* pitch) {
            ImGui::SliderFloat("Yaw", yaw, -180.0f, 180.0f);
            ImGui::SliderFloat("Pitch", pitch, -89.9f, 89.9f);
            const glm::vec3 d = directionFromAngles(*yaw, *pitch);
            ImGui::TextDisabled("direction = (%.2f, %.2f, %.2f)", d.x, d.y, d.z);
        }

        void drawLightsWindow() {
            const float width = 340.0f;
            ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - width - 10.0f, 10.0f),
                ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(width, 480.0f), ImGuiCond_FirstUseEver);
            ImGui::Begin("Lights");

            if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("Show lights", &show_lights);
                ImGui::SliderFloat("Sphere size", &light_sphere_size, 0.02f, 0.3f);
                ImGui::SliderFloat("Cone length", &spot_cone_length, 0.2f, 5.0f);
            }

            if (ImGui::CollapsingHeader("Ambient light", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::PushID("ambient");
                ImGui::ColorEdit3("Color", &ambient_light.color.x);
                ImGui::SliderFloat("Intensity", &ambient_light.intensity, 0.0f, 1.0f);
                ImGui::PopID();
            }

            if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::PushID("directional");
                ImGui::Checkbox("Enabled", &directional_light.enabled);
                directionEditor(&directional_light.yaw, &directional_light.pitch);
                ImGui::ColorEdit3("Color", &directional_light.color.x);
                ImGui::SliderFloat("Intensity", &directional_light.intensity, 0.0f, 5.0f);
                ImGui::PopID();
            }


            if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::PushID("point");
                if (ImGui::Button("Add point light")) {
                    point_lights.push_back(PointLightSettings{});
                }
                ImGui::SameLine();
                ImGui::Text("count: %d", int(point_lights.size()));

                int remove_index = -1;
                for (size_t i = 0; i < point_lights.size(); ++i) {
                    PointLightSettings& light = point_lights[i];

                    ImGui::PushID(int(i));
                    if (ImGui::TreeNode("light", "Point light #%d", int(i) + 1)) {
                        ImGui::Checkbox("Enabled", &light.enabled);
                        ImGui::SameLine();
                        if (ImGui::SmallButton("Remove")) {
                            remove_index = int(i);
                        }
                        ImGui::DragFloat3("Position", &light.position.x, 0.05f);
                        ImGui::ColorEdit3("Color", &light.color.x);
                        ImGui::DragFloat("Intensity", &light.intensity, 0.05f, 0.0f, 500.0f);
                        ImGui::Checkbox("Orbit", &light.orbit);
                        if (light.orbit) {
                            ImGui::SliderFloat("Orbit speed", &light.orbit_speed, -3.0f, 3.0f);
                        }
                        ImGui::TreePop();
                    }
                    ImGui::PopID();
                }

                if (remove_index >= 0) {
                    point_lights.erase(point_lights.begin() + remove_index);
                }
                ImGui::PopID();
            }


            if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::PushID("spot");
                if (ImGui::Button("Add spot light")) {
          
                    SpotLightSettings light;
                    light.position = camera.position;
                    light.yaw = camera.yaw;
                    light.pitch = camera.pitch;
                    spot_lights.push_back(light);
                }
                ImGui::SameLine();
                ImGui::Text("count: %d", int(spot_lights.size()));

                int remove_index = -1;
                for (size_t i = 0; i < spot_lights.size(); ++i) {
                    SpotLightSettings& light = spot_lights[i];

                    ImGui::PushID(int(i));
                    if (ImGui::TreeNode("light", "Spot light #%d", int(i) + 1)) {
                        ImGui::Checkbox("Enabled", &light.enabled);
                        ImGui::SameLine();
                        if (ImGui::SmallButton("Remove")) {
                            remove_index = int(i);
                        }
                        ImGui::Checkbox("Attach to camera (flashlight)", &light.attach_to_camera);
                        if (!light.attach_to_camera) {
                            ImGui::DragFloat3("Position", &light.position.x, 0.05f);
                            directionEditor(&light.yaw, &light.pitch);
                        }
                        ImGui::ColorEdit3("Color", &light.color.x);
                        ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 1000.0f);

                       
                        ImGui::SliderFloat("Inner angle", &light.inner_angle, 0.0f, 89.0f);
                        ImGui::SliderFloat("Outer angle", &light.outer_angle, 0.0f, 89.0f);
                        light.outer_angle = std::max(light.outer_angle, light.inner_angle);

                        ImGui::TreePop();
                    }
                    ImGui::PopID();
                }

                if (remove_index >= 0) {
                    spot_lights.erase(spot_lights.begin() + remove_index);
                }
                ImGui::PopID();
            }

            if (ImGui::Button("Reset lights")) {
                resetLights();
            }

            ImGui::End();
        }


        void addLightSphere(const glm::vec3& position, const glm::vec3& color) {
            LightPushConstants sphere{};
            sphere.model = glm::translate(glm::mat4(1.0f), position);
            sphere.model = glm::scale(sphere.model, glm::vec3(light_sphere_size));
            sphere.color = glm::vec4(color, 1.0f);
            light_spheres.push_back(sphere);
        }

        void addLine(const glm::vec3& a, const glm::vec3& b, const glm::vec3& color) {
            line_vertices.push_back({ a, glm::vec3(0.0f), color });
            line_vertices.push_back({ b, glm::vec3(0.0f), color });
        }

        void addSpotCone(const glm::vec3& position, const glm::vec3& direction,
                         float inner_angle, float outer_angle) {
            const glm::vec3 white = glm::vec3(1.0f);
            const glm::vec3 grey = glm::vec3(0.5f);

            const glm::vec3 helper = std::fabs(direction.y) < 0.99f
                ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
            const glm::vec3 side = glm::normalize(glm::cross(direction, helper));
            const glm::vec3 up = glm::cross(side, direction);

            const glm::vec3 center = position + direction * spot_cone_length;
            const float outer_radius = spot_cone_length * std::tan(glm::radians(std::min(outer_angle, 89.0f)));
            const float inner_radius = spot_cone_length * std::tan(glm::radians(std::min(inner_angle, 89.0f)));

            const int segments = 32;
            for (int i = 0; i < segments; ++i) {
                const float a0 = 2.0f * PI * float(i) / float(segments);
                const float a1 = 2.0f * PI * float(i + 1) / float(segments);
                const glm::vec3 d0 = side * std::cos(a0) + up * std::sin(a0);
                const glm::vec3 d1 = side * std::cos(a1) + up * std::sin(a1);

                addLine(center + d0 * outer_radius, center + d1 * outer_radius, white);
                if (inner_radius > 0.0f) {
                    addLine(center + d0 * inner_radius, center + d1 * inner_radius, grey);
                }
                if (i % 4 == 0) {
                    addLine(position, center + d0 * outer_radius, white);
                }
            }

            addLine(position, position + direction * (spot_cone_length * 1.25f), white);
        }

        void prepareFrame(float delta) {
            const auto& extent = graphics::internal::context.swapchain_extent;
            const float aspect = float(extent.width) / float(std::max(extent.height, 1u));

  
            glm::mat4 camera_model = glm::translate(glm::mat4(1.0f), camera.position);
            camera_model = glm::rotate(camera_model, glm::radians(-camera.yaw), glm::vec3(0, 1, 0));
            camera_model = glm::rotate(camera_model, glm::radians(camera.pitch), glm::vec3(1, 0, 0));
            const glm::mat4 view = glm::inverse(camera_model);

            glm::mat4 proj;
            if (camera.projection_mode == 0) {
                proj = glm::perspective(glm::radians(camera.fov), aspect, camera.near_plane, camera.far_plane);
            } else {
                const float s = camera.ortho_size;
                proj = glm::ortho(-s * aspect, s * aspect, -s, s, camera.near_plane, camera.far_plane);
            }
            proj[1][1] *= -1.0f;

            const glm::vec3 camera_forward = directionFromAngles(camera.yaw, camera.pitch);
            const glm::vec3 camera_right = glm::normalize(glm::cross(camera_forward, glm::vec3(0, 1, 0)));


            scene_uniforms = {};
            scene_uniforms.view_projection = proj * view;
            scene_uniforms.camera_position = camera.position;
            scene_uniforms.directional_light_direction =
                directionFromAngles(directional_light.yaw, directional_light.pitch);
            scene_uniforms.directional_light_color = directional_light.enabled
                ? directional_light.color * directional_light.intensity : glm::vec3(0.0f);
            scene_uniforms.ambient_light = ambient_light.color * ambient_light.intensity;

            light_spheres.clear();
            line_vertices.clear();

            point_light_data.clear();
            for (PointLightSettings& light : point_lights) {

                if (light.orbit && is_playing) {
                    const float angle = light.orbit_speed * anim_speed * delta;
                    const float c = std::cos(angle);
                    const float s = std::sin(angle);
                    const float x = light.position.x;
                    const float z = light.position.z;
                    light.position.x = c * x - s * z;
                    light.position.z = s * x + c * z;
                }

                if (!light.enabled) {
                    continue; 
                }

                PointLight gpu{};
                gpu.position = light.position;
                gpu.intensity = light.intensity;
                gpu.color = light.color;
                point_light_data.push_back(gpu);

                if (show_lights) {
                    addLightSphere(light.position, light.color);
                }
            }


            spot_light_data.clear();
            for (SpotLightSettings& light : spot_lights) {
                if (light.attach_to_camera) {

                    light.position = camera.position + camera_right * 0.25f - glm::vec3(0.0f, 0.2f, 0.0f);
                    light.yaw = camera.yaw;
                    light.pitch = camera.pitch;
                }

                if (!light.enabled) {
                    continue;
                }

                const glm::vec3 direction = directionFromAngles(light.yaw, light.pitch);

                SpotLight gpu{};
                gpu.position = light.position;
                gpu.intensity = light.intensity;
                gpu.color = light.color;
                gpu.direction = direction;
                gpu.inner_angle = std::cos(glm::radians(light.inner_angle));
                gpu.outer_angle = std::cos(glm::radians(light.outer_angle));
                spot_light_data.push_back(gpu);

                if (show_lights && !light.attach_to_camera) {
                    addLightSphere(light.position, light.color);
                    addSpotCone(light.position, direction, light.inner_angle, light.outer_angle);
                }
            }

            scene_uniforms.point_light_count = uint32_t(point_light_data.size());
            scene_uniforms.spot_light_count = uint32_t(spot_light_data.size());

           
            for (SceneObject& object : objects) {
                glm::vec3 position = object.position;
                float extra_rotation = 0.0f;

                if (object.animate) {
                    const float phase = anim_angle + object.phase;
                    position.x += anim_radius * std::sin(phase);
                    position.z += anim_radius * std::sin(phase * 2.0f) * 0.5f;
                    position.y += anim_height * std::sin(phase * 3.0f);
                    extra_rotation = phase * 2.0f;
                }

                glm::mat4 model = glm::translate(glm::mat4(1.0f), position);
                model = glm::rotate(model, glm::radians(object.rotation.x), glm::vec3(1, 0, 0));
                model = glm::rotate(model, glm::radians(object.rotation.y) + extra_rotation, glm::vec3(0, 1, 0));
                model = glm::rotate(model, glm::radians(object.rotation.z), glm::vec3(0, 0, 1));
                model = glm::scale(model, object.scale);

                object.uniforms = {};
                object.uniforms.model = model;
                object.uniforms.albedo = object.albedo;
                object.uniforms.specular = object.specular;
                object.uniforms.alpha = std::max(object.alpha, 1.0f);
            }
        }

    } // namespace


    bool initialize() {
        resetLights();

        if (!createMesh(makeCube(), cube_mesh) ||
            !createMesh(makeSphere(10, 16), sphere_mesh)) {
            return false;
        }


        objects.push_back(makeObject("Cube 1 (center)", &cube_mesh, glm::vec3(0.0f), 1.0f,
            glm::vec3(0.2f, 0.8f, 0.5f), glm::vec3(1.0f), 64.0f, 0.0f));
        objects.push_back(makeObject("Cube 2 (left)", &cube_mesh, glm::vec3(-2.0f, 0.0f, 0.0f), 0.8f,
            glm::vec3(1.0f, 0.5f, 0.2f), glm::vec3(0.5f), 16.0f, 2.0f * PI / 3.0f));
        objects.push_back(makeObject("Cube 3 (right)", &cube_mesh, glm::vec3(2.0f, 0.0f, 0.0f), 0.6f,
            glm::vec3(0.2f, 0.5f, 1.0f), glm::vec3(1.0f, 0.95f, 0.8f), 200.0f, 4.0f * PI / 3.0f));


        if (!createBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr,
            scene_uniform_buffer)) {
            return false;
        }
        for (SceneObject& object : objects) {
            if (!createBuffer(sizeof(ModelUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr,
                object.uniform_buffer)) {
                return false;
            }
        }
        ensureBufferSize(point_light_buffer, sizeof(PointLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        ensureBufferSize(spot_light_buffer, sizeof(SpotLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        ensureBufferSize(line_vertex_buffer, 2 * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        if (!createDescriptors()) {
            return false;
        }

        vk_pipeline = createPipeline("shader.vert.spv", "shader.frag.spv", vk_pipeline_layout,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_CULL_MODE_BACK_BIT);
        vk_light_pipeline = createPipeline("light.vert.spv", "light.frag.spv", vk_light_pipeline_layout,
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_CULL_MODE_NONE);
        vk_line_pipeline = createPipeline("light.vert.spv", "light.frag.spv", vk_light_pipeline_layout,
            VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_CULL_MODE_NONE);

        return vk_pipeline != VK_NULL_HANDLE
            && vk_light_pipeline != VK_NULL_HANDLE
            && vk_line_pipeline != VK_NULL_HANDLE;
    }

    void shutdown() {
        auto& context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);

        vkDestroyPipeline(context.device, vk_line_pipeline, nullptr);
        vkDestroyPipeline(context.device, vk_light_pipeline, nullptr);
        vkDestroyPipeline(context.device, vk_pipeline, nullptr);
        vkDestroyPipelineLayout(context.device, vk_light_pipeline_layout, nullptr);
        vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);

  
        vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
        vkDestroyDescriptorSetLayout(context.device, vk_model_descriptor_set_layout, nullptr);
        vkDestroyDescriptorSetLayout(context.device, vk_scene_descriptor_set_layout, nullptr);

        for (SceneObject& object : objects) {
            destroyBuffer(object.uniform_buffer);
        }
        destroyBuffer(line_vertex_buffer);
        destroyBuffer(spot_light_buffer);
        destroyBuffer(point_light_buffer);
        destroyBuffer(scene_uniform_buffer);

        destroyMesh(sphere_mesh);
        destroyMesh(cube_mesh);
    }


    void update(double time) {
        static double last_time = time;
        const float delta = std::min(float(time - last_time), 0.1f);
        last_time = time;

        if (is_playing) {
            anim_angle += delta * anim_speed;
            if (anim_angle > 2.0f * PI) anim_angle -= 2.0f * PI;
            if (anim_angle < 0.0f) anim_angle += 2.0f * PI;
        }

        drawSceneWindow();
        drawLightsWindow();
        prepareFrame(delta);
    }


    void render(const graphics::internal::FrameData& fd) {

        if (ensureBufferSize(point_light_buffer,
            std::max<size_t>(point_light_data.size(), 1) * sizeof(PointLight),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
            writeBufferDescriptor(vk_scene_descriptor_set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                point_light_buffer.buffer, VK_WHOLE_SIZE);
        }
        if (ensureBufferSize(spot_light_buffer,
            std::max<size_t>(spot_light_data.size(), 1) * sizeof(SpotLight),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
            writeBufferDescriptor(vk_scene_descriptor_set, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                spot_light_buffer.buffer, VK_WHOLE_SIZE);
        }
        ensureBufferSize(line_vertex_buffer,
            std::max<size_t>(line_vertices.size(), 2) * sizeof(Vertex),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        std::memcpy(scene_uniform_buffer.mapped, &scene_uniforms, sizeof(SceneUniforms));
        if (!point_light_data.empty()) {
            std::memcpy(point_light_buffer.mapped, point_light_data.data(),
                point_light_data.size() * sizeof(PointLight));
        }
        if (!spot_light_data.empty()) {
            std::memcpy(spot_light_buffer.mapped, spot_light_data.data(),
                spot_light_data.size() * sizeof(SpotLight));
        }
        if (!line_vertices.empty()) {
            std::memcpy(line_vertex_buffer.mapped, line_vertices.data(),
                line_vertices.size() * sizeof(Vertex));
        }
        for (SceneObject& object : objects) {
            std::memcpy(object.uniform_buffer.mapped, &object.uniforms, sizeof(ModelUniform));
        }


        vkResetCommandBuffer(fd.command_buffer, 0);

        const VkCommandBufferBeginInfo command_buffer_begin = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };

        vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin);

        const VkClearValue clear_values[] = {
            {.color = {.float32 = { 0.02f, 0.02f, 0.03f, 1.0f } } },
            {.depthStencil = { 1.0f, 0 } },
        };

        const VkRenderPassBeginInfo render_pass_begin = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = graphics::internal::context.render_pass,
            .framebuffer = fd.framebuffer,
            .renderArea = {.extent = graphics::internal::context.swapchain_extent },
            .clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
            .pClearValues = clear_values,
        };

        vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

        const VkViewport viewport = {
            .x = 0.0f, .y = 0.0f,
            .width = float(graphics::internal::context.swapchain_extent.width),
            .height = float(graphics::internal::context.swapchain_extent.height),
            .minDepth = 0.0f, .maxDepth = 1.0f,
        };

        const VkRect2D scissor = {
            .extent = graphics::internal::context.swapchain_extent,
        };

        vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

        const VkDeviceSize offset = 0;

        vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            vk_pipeline_layout, 0, 1, &vk_scene_descriptor_set, 0, nullptr);

        for (const SceneObject& object : objects) {
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                vk_pipeline_layout, 1, 1, &object.descriptor_set, 0, nullptr);

            vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &object.mesh->vertex_buffer.buffer, &offset);
            vkCmdBindIndexBuffer(fd.command_buffer, object.mesh->index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(fd.command_buffer, object.mesh->index_count, 1, 0, 0, 0);
        }

        if (show_lights) {
            vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_light_pipeline);
            vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                vk_light_pipeline_layout, 0, 1, &vk_scene_descriptor_set, 0, nullptr);

            vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &sphere_mesh.vertex_buffer.buffer, &offset);
            vkCmdBindIndexBuffer(fd.command_buffer, sphere_mesh.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

            for (const LightPushConstants& sphere : light_spheres) {
                vkCmdPushConstants(fd.command_buffer, vk_light_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                    0, sizeof(LightPushConstants), &sphere);
                vkCmdDrawIndexed(fd.command_buffer, sphere_mesh.index_count, 1, 0, 0, 0);
            }

            if (!line_vertices.empty()) {
                vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_line_pipeline);

                const LightPushConstants identity = { glm::mat4(1.0f), glm::vec4(1.0f) };
                vkCmdPushConstants(fd.command_buffer, vk_light_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                    0, sizeof(LightPushConstants), &identity);

                vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &line_vertex_buffer.buffer, &offset);
                vkCmdDraw(fd.command_buffer, uint32_t(line_vertices.size()), 1, 0, 0);
            }
        }

        vkCmdEndRenderPass(fd.command_buffer);
        vkEndCommandBuffer(fd.command_buffer);
    }

} // namespace application
