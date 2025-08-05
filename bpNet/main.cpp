#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>    // 用于高精度计时
#include <limits>    // 用于清空输入缓冲区
#include <Windows.h> // 用于设置控制台编码
#include "bpNet.h"

// 函数声明
bool loadDataFromFile(const std::string& filename, int num_inputs, int num_outputs,
                      std::vector<std::vector<double>>& inputs, 
                      std::vector<std::vector<double>>& targets);

int main() {
    // 设置控制台输出编码为UTF-8
    SetConsoleOutputCP(65001);

    // --- 1. 定义网络结构和数据文件 ---
    const int input_nodes = 2;
    const int hidden_nodes = 4;
    const int output_nodes = 1;
    const double learning_rate = 0.4;
    const std::string train_filename = "data.txt";

    // --- 2. 加载训练数据 ---
    std::vector<std::vector<double>> train_inputs;
    std::vector<std::vector<double>> train_targets;
    std::cout << "--- 数据加载阶段 ---" << std::endl;
    std::cout << "正在从文件 \"" << train_filename << "\" 加载训练数据..." << std::endl;
    if (!loadDataFromFile(train_filename, input_nodes, output_nodes, train_inputs, train_targets)) {
        std::cerr << "数据加载失败，程序即将退出。" << std::endl;
        std::cin.get();
        return 1;
    }
    std::cout << "训练数据加载成功！共加载了 " << train_inputs.size() << " 条样本。" << std::endl;

    // --- 3. 获取用户输入参数 ---
    double target_error;
    int max_epochs;
    std::cout << "\n--- 训练参数设置 ---" << std::endl;
    std::cout << "请输入您的目标平均损失 (例如 0.001): ";
    while (!(std::cin >> target_error) || target_error <= 0) {
        std::cout << "输入无效，请输入一个大于0的数字: ";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    std::cout << "请输入最大训练轮数 (例如 50000): ";
    while (!(std::cin >> max_epochs) || max_epochs <= 0) {
        std::cout << "输入无效，请输入一个正整数: ";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    // --- 4. 创建网络并开始训练 ---
    bpNet network(input_nodes, hidden_nodes, output_nodes, learning_rate);
    std::cout << "\n--- 网络训练阶段 ---" << std::endl;
    std::cout << "参数已设定。按回车键开始训练..." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();

    auto start_time = std::chrono::high_resolution_clock::now(); // 开始计时

    bool target_achieved = false;
    int final_epoch = 0;
    double current_loss = 1.0;

    for (int epoch = 1; epoch <= max_epochs; ++epoch) {
        double epoch_loss = 0;
        for (size_t i = 0; i < train_inputs.size(); ++i) {
            network.train(train_inputs[i], train_targets[i]);
            epoch_loss += network.get_loss(train_targets[i]);
        }
        current_loss = epoch_loss / train_inputs.size();
        final_epoch = epoch;

        // 【可视化核心】使用\r让光标回到行首，实现单行刷新
        std::cout << "\r训练进度: 轮次 " << std::setw(6) << epoch << "/" << max_epochs
                  << " | 当前平均损失: " << std::fixed << std::setprecision(8) << current_loss << std::flush;

        // 检查是否达到目标误差
        if (current_loss <= target_error) {
            target_achieved = true;
            break; // 达到目标，跳出循环
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now(); // 结束计时
    std::chrono::duration<double> training_duration = end_time - start_time;

    std::cout << std::endl; // 训练结束后换行，保留最后一条进度信息

    // --- 5. 输出最终训练结果 ---
    std::cout << "\n--- 训练结果总结 ---" << std::endl;
    if (target_achieved) {
        std::cout << "训练成功！已达到目标误差。" << std::endl;
        std::cout << " - 目标误差: " << target_error << std::endl;
        std::cout << " - 最终误差: " << current_loss << std::endl;
        std::cout << " - 所用轮次: " << final_epoch << " / " << max_epochs << std::endl;
    } else {
        std::cout << "训练停止。已达到最大训练轮数但未达到目标误差。" << std::endl;
        std::cout << " - 目标误差: " << target_error << std::endl;
        std::cout << " - 最终误差: " << current_loss << std::endl;
        std::cout << " - 所用轮次: " << final_epoch << " / " << max_epochs << std::endl;
    }
    std::cout << " - 训练耗时: " << std::fixed << std::setprecision(3) << training_duration.count() << " 秒" << std::endl;

    // --- 6. 展示最终预测效果 ---
    std::cout << "\n--- 网络预测效果 ---" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < train_inputs.size(); ++i) {
        std::vector<double> prediction = network.predict(train_inputs[i]);
        std::cout << "输入: {" << train_inputs[i][0] << ", " << train_inputs[i][1] << "} "
                  << "-> 预测值: " << prediction[0]
                  << " (期望值: " << train_targets[i][0] << ")" << std::endl;
    }

    // --- 让程序暂停 ---
    std::cout << "\n按回车键退出程序...";
    std::cin.get();

    return 0;
}


// 数据加载函数
bool loadDataFromFile(const std::string& filename, int num_inputs, int num_outputs,
                      std::vector<std::vector<double>>& inputs, 
                      std::vector<std::vector<double>>& targets) {
    inputs.clear();
    targets.clear();
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "错误：无法打开数据文件 \"" << filename << "\"" << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::vector<double> input_sample;
        std::vector<double> target_sample;
        double value;
        for (int i = 0; i < num_inputs; ++i) {
            if (!(ss >> value)) continue;
            input_sample.push_back(value);
        }
        for (int i = 0; i < num_outputs; ++i) {
            if (!(ss >> value)) continue;
            target_sample.push_back(value);
        }
        if (input_sample.size() == num_inputs && target_sample.size() == num_outputs) {
            inputs.push_back(input_sample);
            targets.push_back(target_sample);
        }
    }
    file.close();
    return true;

}
