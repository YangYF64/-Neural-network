#ifndef BPNET_H
#define BPNET_H

#include <vector>
#include <random> // 使用现代C++的随机数库

class bpNet {
public:
    // --- 公共接口 ---

    /*
     * @brief 构造函数
     * @param i_node 输入层节点数
     * @param h_node 隐藏层节点数
     * @param o_node 输出层节点数
     * @param learn_rate 学习率
     */
    bpNet(int i_node, int h_node, int o_node, double learn_rate = 0.3);

    /*
     * @brief 使用一组数据进行训练（一次前向传播和一次反向传播）
     * @param i_data 输入数据
     * @param t_data 期望的输出数据（标签）
     */
    void train(const std::vector<double>& i_data, const std::vector<double>& t_data);

    /*
     * @brief 使用训练好的网络进行预测
     * @param i_data 输入数据
     * @return 网络的预测输出
     */
    std::vector<double> predict(const std::vector<double>& i_data);

    /*
     * @brief 计算当前输出相对于目标值的均方误差(MSE)
     * @param t_data 期望的输出数据（标签）
     * @return 均方误差值
     */
    double get_loss(const std::vector<double>& t_data) const;


private:
    // --- 私有成员变量 ---
    int i_node, h_node, o_node;
    double learn_rate;

    // 权重和偏置
    std::vector<std::vector<double>> ih_w; // 输入层 -> 隐藏层 权重
    std::vector<std::vector<double>> ho_w; // 隐藏层 -> 输出层 权重
    std::vector<double> h_b;              // 隐藏层偏置
    std::vector<double> o_b;              // 输出层偏置

    // 各层输入/输出（在计算过程中复用）
    std::vector<double> i_data;           // 当前输入数据
    std::vector<double> h_out;            // 隐藏层输出
    std::vector<double> o_out;            // 输出层输出

    // 梯度（在计算过程中复用）
    std::vector<double> h_grad;           // 隐藏层梯度
    std::vector<double> o_grad;           // 输出层梯度

    // C++11 随机数生成器
    std::mt19937 gen;                     // 随机数引擎
    std::uniform_real_distribution<double> dis; // -1到1的均匀分布

private:
    // --- 私有辅助函数 ---

    /*
     * @brief 初始化权重和偏置
     */
    void init();

    /*
     * @brief 前向传播过程
     * @param input_data 输入数据
     */
    void forward_propa(const std::vector<double>& input_data);

    /*
     * @brief 反向传播过程
     * @param target_data 期望的输出数据
     */
    void back_propa(const std::vector<double>& target_data);

    /*
     * @brief Sigmoid激活函数
     * @param z 净输入
     * @return 激活后的值
     */
    double sigmoid(double z) const;
};

#endif // BPNET_H