#include "ModelInterpolator.h"

void ModelInterpolator::update_weights(float alpha) {
    // Desactivar el cálculo de gradientes para la actualización
    torch::NoGradGuard no_grad;

    // Obtener listas de parámetros de los modelos
    auto params_a = model_a.parameters();
    auto params_b = model_b.parameters();
    auto params_active = active_model.parameters();

    // Usar iteradores directamente
    auto it_a = params_a.begin();
    auto it_b = params_b.begin();
    auto it_active = params_active.begin();

    // Iterar sobre todos los parámetros simultáneamente
    while (it_a != params_a.end() && it_b != params_b.end() && it_active != params_active.end()) {
        const auto& p_a = *it_a;
        const auto& p_b = *it_b;
        auto p_active = *it_active;  // Obtener copia del tensor
        
        // Calcular el tensor interpolado usando LERP
        torch::Tensor interpolated_p = torch::lerp(p_a, p_b, alpha);
        
        // Copiar los nuevos valores al parámetro del modelo activo
        p_active.copy_(interpolated_p);

        ++it_a;
        ++it_b;
        ++it_active;
    }
}