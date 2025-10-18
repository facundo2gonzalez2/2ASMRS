#include <torch/script.h>

class ModelInterpolator {
public:
    ModelInterpolator(const std::string& path_a, const std::string& path_b) {
        // Cargar los modelos de origen desde el disco
        model_a = torch::jit::load(path_a);
        model_b = torch::jit::load(path_b);

        // El modelo activo es inicialmente una copia del modelo A
        active_model = torch::jit::load(path_a);
        active_model.eval(); // Poner en modo de evaluación
    }

    // Método para realizar la inferencia
    torch::Tensor forward(torch::Tensor input) {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        return active_model.forward(inputs).toTensor();
    }

    // Método para actualizar los pesos del modelo activo
    void update_weights(float alpha);

private:
    torch::jit::script::Module model_a;
    torch::jit::script::Module model_b;
    torch::jit::script::Module active_model;
};