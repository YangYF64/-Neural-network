#include "bpNet.h"
#include <cmath>     // For std::exp
#include <numeric>   // For std::inner_product if needed, not used here but good practice
#include <algorithm> // For std::transform if needed

// --- 构造函数 ---
bpNet::bpNet(int i_node, int h_node, int o_node, double learn_rate) :
    // 使用成员初始化列表进行初始化
    i_node(i_node),
    h_node(h_node),
    o_node(o_node),
    learn_rate(learn_rate),
    gen(std::random_device{}()), // 使用硬件熵作为种子
    dis(-1.0, 1.0)               // 初始化分布范围为[-1.0, 1.0]
{
    // 为所有vector分配空间
    ih_w.resize(i_node, std::vector<double>(h_node));
    ho_w.resize(h_node, std::vector<double>(o_node));
    h_b.resize(h_node);
    o_b.resize(o_node);
    
    this->i_data.resize(i_node);
    h_out.resize(h_node);
    o_out.resize(o_node);

    h_grad.resize(h_node);
    o_grad.resize(o_node);
    
    // 调用初始化函数来填充权重和偏置
    init();
}

// --- 初始化函数 ---
void bpNet::init() {
    // 初始化输入层到隐藏层的权重
    for (int i = 0; i < i_node; ++i) {
        for (int j = 0; j < h_node; ++j) {
            ih_w[i][j] = dis(gen); // 使用新的随机数生成方式
        }
    }
    // 初始化隐藏层的偏置
    for (int i = 0; i < h_node; ++i) {
        h_b[i] = dis(gen);
    }
    // 初始化隐藏层到输出层的权重
    for (int i = 0; i < h_node; ++i) {
        for (int j = 0; j < o_node; ++j) {
            ho_w[i][j] = dis(gen);
        }
    }
    // 初始化输出层的偏置
    for (int i = 0; i < o_node; ++i) {
        o_b[i] = dis(gen);
    }
}

// --- Sigmoid 激活函数 ---
double bpNet::sigmoid(double z) const {
    return 1.0 / (1.0 + std::exp(-z));
}

// --- 前向传播 ---
void bpNet::forward_propa(const std::vector<double>& input_data) {
    this->i_data = input_data;

    // 计算隐藏层输出
    for (int i = 0; i < h_node; ++i) {
        double z = 0.0;
        for (int j = 0; j < i_node; ++j) {
            z += this->i_data[j] * ih_w[j][i];
        }
        z += h_b[i];
        h_out[i] = sigmoid(z);
    }

    // 计算输出层输出
    for (int i = 0; i < o_node; ++i) {
        double z = 0.0;
        for (int j = 0; j < h_node; ++j) {
            z += h_out[j] * ho_w[j][i];
        }
        z += o_b[i];
        o_out[i] = sigmoid(z);
    }
}

// --- 反向传播 ---
void bpNet::back_propa(const std::vector<double>& target_data) {
    // 1. 计算输出层梯度
    for (int i = 0; i < o_node; ++i) {
        o_grad[i] = (target_data[i] - o_out[i]) * o_out[i] * (1.0 - o_out[i]);
    }

    // 2. 计算隐藏层梯度
    for (int i = 0; i < h_node; ++i) {
        double error_sum = 0.0;
        for (int j = 0; j < o_node; ++j) {
            error_sum += o_grad[j] * ho_w[i][j];
        }
        h_grad[i] = error_sum * h_out[i] * (1.0 - h_out[i]);
    }

    // 3. 更新隐藏层到输出层的权重和偏置
    for (int i = 0; i < h_node; ++i) {
        for (int j = 0; j < o_node; ++j) {
            ho_w[i][j] += learn_rate * o_grad[j] * h_out[i];
        }
    }
    for (int i = 0; i < o_node; ++i) {
        o_b[i] += learn_rate * o_grad[i];
    }

    // 4. 更新输入层到隐藏层的权重和偏置
    for (int i = 0; i < i_node; ++i) {
        for (int j = 0; j < h_node; ++j) {
            ih_w[i][j] += learn_rate * h_grad[j] * this->i_data[i];
        }
    }
    for (int i = 0; i < h_node; ++i) {
        h_b[i] += learn_rate * h_grad[i];
    }
}

// --- 训练函数 (公共接口) ---
void bpNet::train(const std::vector<double>& i_data, const std::vector<double>& t_data) {
    forward_propa(i_data);
    back_propa(t_data);
}

// --- 预测函数 (公共接口) ---
std::vector<double> bpNet::predict(const std::vector<double>& i_data) {
    forward_propa(i_data);
    return o_out; // 返回前向传播后的结果
}

// --- 获取损失函数 (公共接口) ---
double bpNet::get_loss(const std::vector<double>& t_data) const {
    double loss = 0.0;
    for (int i = 0; i < o_node; ++i) {
        loss += (t_data[i] - o_out[i]) * (t_data[i] - o_out[i]);
    }
    return loss / 2.0;
}
