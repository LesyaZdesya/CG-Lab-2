#version 450 core

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;
layout(location = 2) in vec3 in_color;

layout(location = 0) out vec4 out_color;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    vec3 directional_light_direction; 
    vec3 directional_light_color;     
    vec3 ambient_light;               
    uint point_light_count;
    uint spot_light_count;
} scene_uniforms;

layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec3 albedo;
    vec3 specular;
    float alpha;
} model_uniforms;


struct PointLight {
    vec3 position;
    float intensity;
    vec3 color;
};

struct SpotLight {
    vec3 position;
    float intensity;
    vec3 color;
    float inner_angle; 
    vec3 direction;    
    float outer_angle; 
};


layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};


vec3 blinnPhong(vec3 n, vec3 v, vec3 l, vec3 light, vec3 albedo) {
    float n_dot_l = max(dot(n, l), 0.0);
    vec3 h = normalize(v + l); // половинчатый вектор
    float n_dot_h = max(dot(n, h), 0.0);
    return (albedo + model_uniforms.specular * pow(n_dot_h, model_uniforms.alpha)) * n_dot_l * light;
}

void main() {
    vec3 n = normalize(in_world_normal);
    vec3 v = normalize(scene_uniforms.camera_position - in_world_position);
    vec3 albedo = model_uniforms.albedo * in_color;

    vec3 color = scene_uniforms.ambient_light * albedo;


    vec3 l_dir = normalize(-scene_uniforms.directional_light_direction);
    color += blinnPhong(n, v, l_dir, scene_uniforms.directional_light_color, albedo);


    for (uint i = 0u; i < scene_uniforms.point_light_count; ++i) {
        vec3 to_light = point_lights[i].position - in_world_position;
        float distance_sq = max(dot(to_light, to_light), 0.0001);
        vec3 l = to_light / sqrt(distance_sq);

        vec3 light = point_lights[i].color * point_lights[i].intensity / distance_sq;
        color += blinnPhong(n, v, l, light, albedo);
    }


    for (uint i = 0u; i < scene_uniforms.spot_light_count; ++i) {
        vec3 to_light = spot_lights[i].position - in_world_position;
        float distance_sq = max(dot(to_light, to_light), 0.0001);
        vec3 l = to_light / sqrt(distance_sq);


        float cos_theta = dot(-l, spot_lights[i].direction);


        float epsilon = max(spot_lights[i].inner_angle - spot_lights[i].outer_angle, 0.0001);
        float edge = clamp((cos_theta - spot_lights[i].outer_angle) / epsilon, 0.0, 1.0);
        edge = smoothstep(0.0, 1.0, edge);

        vec3 light = spot_lights[i].color * spot_lights[i].intensity * edge / distance_sq;
        color += blinnPhong(n, v, l, light, albedo);
    }

    out_color = vec4(color, 1.0);
}
