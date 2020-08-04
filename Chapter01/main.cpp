#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include<opencv2/opencv.hpp>
#include <vector>
#include <string>

int main() {
    try {
        //读取图片
        cv::Mat image = cv::imread("7.jpg", cv::IMREAD_GRAYSCALE);

        //图片数据转换为tensor数据，并把数据转换为float
        torch::Tensor tensor_image = torch::from_blob(image.data, {1, image.rows * image.cols}, torch::kByte).toType(
                torch::kFloat);

        tensor_image /= 255.;
        std::cout << tensor_image.sizes() << std::endl;

        //读取模型
        auto module = torch::jit::load("mnist.pt");
//
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor_image);
//
        auto rst = module.forward(inputs).toTensor();
//
        std::cout << rst << std::endl;
        std::cout << torch::argmax(rst, 1) << std::endl;

    } catch (const c10::Error &e) {
        std::cerr << e.what();
        return -1;
    }

    return 0;
}