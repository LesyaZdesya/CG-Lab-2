#version 450 core

layout(location = 0) in vec3 in_position;
layout(location = 2) in vec3 in_color;

layout(location = 0) out vec3 out_color;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    vec3 directional_light_direction;
    vec3 directional_light_color;
    vec3 ambient_light;
    uint point_light_count;
    uint spot_light_count;
} scene_uniforms;

layout(push_constant) uniform LightPushConstants {
    mat4 model;
    vec4 color;
} push_constants;

void main() {
    gl_Position = scene_uniforms.view_projection * push_constants.model * vec4(in_position, 1.0);
    out_color = in_color * push_constants.color.rgb;
}
