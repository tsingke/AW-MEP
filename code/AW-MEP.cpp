/**************************************************************************
*  Algorithm： AW-MEP 
**************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <float.h> 
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <ctime>
#include <cstdlib>
#include <algorithm> // for std::sort
#include <random>
#define PROBLEM_REGRESSION 0
#define PROBLEM_BINARY_CLASSIFICATION 1
#define MAX_SAMPLES 1000
#define MAX_FEATURES 100
#define NUM_Operators 14  // 更新为12个操作符

// 基本算术操作符
#define ADD_OP -1   // +
#define DIF_OP -2   // -
#define MUL_OP -3   // *
#define DIV_OP -4   // /
#define POW_OP -5   // ^
#define SIN_OP -6   // sin
#define TAN_OP -7   // tan
#define LOG_OP -8   // log
#define EXP_OP -9   // exp
#define SQRT_OP -10 // sqrt
#define ABS_OP -11  // abs
#define NEG_OP -12  // Neg (取相反数)
#define CUBE_OP -13  // 三次方
#define CUBEROOT_OP -14 // 立方根
double initial_temperature = 100.0;  // 初始温度
double cooling_rate = 0.99;  // 降温速率
double min_temperature = 0.01;  // 最低温度
// 操作符字符串映射
char operators_string[NUM_Operators + 1] = "+-*/^stlexaNcu";  // 更新操作符字符串
double operator_weights[NUM_Operators];

// 初始化操作符权重

struct gene {
	int op;
	int a1;
	int a2;
	double fitness;
};

struct chromosome {
	int length = 40;
	double best_fitness;
	int best_index;
	gene* g;
	double* c;
};

struct parameters {
	int size;
	int num;  //迭代次数
	int nc;  //常量个数
	double c_min, c_max;
	double po, pc, pv;
	int problem_type;  //0-regression,1-classification;
};
void allocate_chromosome(chromosome& ch, const parameters& pa)
{
	ch.g = new gene[ch.length];
	if (pa.nc)
		ch.c = new double[pa.nc];// constants
	else
		ch.c = NULL;
}
void delete_chromosome(chromosome& ch)
{
	if (ch.g) {
		delete[] ch.g;
		ch.g = NULL;
	}
	if (ch.c) {
		delete[] ch.c;
		ch.c = NULL;
	}
}
void allocate_traindata(double**& data, double*& target, int num_t, int num_v) {
	if (num_t <= 0 || num_v <= 0) {
		throw std::invalid_argument("Invalid array size");
	}

	target = new double[num_t];
	if (target == NULL) {
		throw std::bad_alloc();  // 抛出内存分配失败异常
	}

	data = new double* [num_t];
	if (data == NULL) {
		delete[] target;  // 清理已分配的内存
		throw std::bad_alloc();  // 抛出内存分配失败异常
	}

	for (int i = 0; i < num_t; i++) {
		data[i] = new double[num_v];
		if (data[i] == NULL) {
			// 清理已分配的内存
			for (int j = 0; j < i; j++) {
				delete[] data[j];
			}
			delete[] data;
			delete[] target;
			throw std::bad_alloc();  // 抛出内存分配失败异常
		}
	}
}
void delete_traindata(double**& data, double*& target, int num_t)
{
	if (data)
		for (int i = 0; i < num_t; i++)
			delete[] data[i];
	delete[] data;
	delete[] target;
}
void allocate_partial_expression_values(double**& expression_value, int num_t, int gene_l)
{// allocate memory for the matrix storing the output of each expression for each training data
	// this is allocated once and then reused, in order to reduce the number of allocations/deletions
	expression_value = new double* [gene_l];
	for (int i = 0; i < gene_l; i++)
		expression_value[i] = new double[num_t];
}
void delete_partial_expression_values(double**& expression_value, int gene_l)
{
	if (expression_value) {
		for (int i = 0; i < gene_l; i++)
			delete[] expression_value[i];
		delete[] expression_value;
	}
}
void copy_individual(chromosome& d, const chromosome& s, const parameters& p)
{
	for (int i = 0; i < s.length; i++)
		d.g[i] = s.g[i];
	for (int i = 0; i < p.nc; i++)
		d.c[i] = s.c[i];
	d.best_fitness = s.best_fitness;
	d.best_index = s.best_index;
}
void print_chromosome(const chromosome& a, const parameters& params, int num_v)
{
	printf("The chromosome is:\n");
	printf("//------------------------------------------\n");
	for (int i = 0; i < params.nc; i++)
		printf("constants[%d] = %lf\n", i, a.c[i]);

	for (int i = 0; i < a.length; i++) {
		if (a.g[i].op < 0) {
			if (a.g[i].op == SIN_OP)
				printf("%d: sin %d\n", i, a.g[i].a1);
			else// binary operators
				printf("%d: %c %d %d\n", i, operators_string[abs(a.g[i].op) - 1], a.g[i].a1, a.g[i].a2);
		}
		else {
			if (a.g[i].op < num_v)
				printf("%d: inputs[%d]\n", i, a.g[i].op);
			else
				printf("%d: constants[%d]\n", i, a.g[i].op - num_v);
		}
	}
	printf("//------------------------------------------\n");
	printf("Best index (output provider) = %d\n", a.best_index);
	printf("Fitness = %lf\n", a.best_fitness);
}

void compute_eval_matrix(const chromosome& c, int l, int num_v, int num_t, const double** traindata, double** eval_matrix) {
#pragma omp parallel for shared(c, traindata, eval_matrix)
	for (int i = 0; i < l; i++) {  // 从上到下读取基因
		bool is_error_case = false; // 用来检查错误情况（如零除）
		int error_count = 0;  // 错误计数器
		switch (c.g[i].op) {
		case ADD_OP: { // +
#pragma omp simd
			for (int k = 0; k < num_t; k++)
				eval_matrix[i][k] = eval_matrix[c.g[i].a1][k] + eval_matrix[c.g[i].a2][k];
			break;
		}
		case DIF_OP: {  // 
#pragma omp simd
			for (int k = 0; k < num_t; k++)
				eval_matrix[i][k] = eval_matrix[c.g[i].a1][k] - eval_matrix[c.g[i].a2][k];
			break;
		}
		case MUL_OP: {  // *
#pragma omp simd
			for (int k = 0; k < num_t; k++)
				eval_matrix[i][k] = eval_matrix[c.g[i].a1][k] * eval_matrix[c.g[i].a2][k];
			break;
		}
		case DIV_OP:
		{
#pragma omp simd
			for (int k = 0; k < num_t; k++) {
				const double denominator = eval_matrix[c.g[i].a2][k];
				if (fabs(denominator) < 1e-6) {
					error_count++;
					eval_matrix[i][k] = 0.0;
				}
				else {
					eval_matrix[i][k] = eval_matrix[c.g[i].a1][k] / denominator;
				}
			}
			// 根据错误数量增加适应度惩罚
			if (error_count > 0) {
#pragma omp critical
				c.g[i].fitness *= (1.0 + error_count / (double)num_t);
			}
			break;
		}
		case POW_OP:  // ^
			for (int k = 0; k < num_t; k++)
				eval_matrix[i][k] = pow(eval_matrix[c.g[i].a1][k], eval_matrix[c.g[i].a2][k]);
			break;
		case SIN_OP:  // sin
			for (int k = 0; k < num_t; k++)
				eval_matrix[i][k] = sin(eval_matrix[c.g[i].a1][k]);
			break;
		case TAN_OP:  // tan
			for (int k = 0; k < num_t; k++) {
				if (eval_matrix[c.g[i].a1][k] == 0) {  // 防止 tan(0) 出现错误
					is_error_case = true;
					eval_matrix[i][k] = 0;  // 赋值为0或者其他安全值
				}
				else {
					eval_matrix[i][k] = tan(eval_matrix[c.g[i].a1][k]);
				}
			}
			if (is_error_case) {
				c.g[i].op = rand() % num_v;   // 基因突变为终结符
				for (int k = 0; k < num_t; k++)
					eval_matrix[i][k] = traindata[k][c.g[i].op];
			}
			break;
		case LOG_OP:  // log
			for (int k = 0; k < num_t; k++) {
				if (eval_matrix[c.g[i].a1][k] > 0) {
					eval_matrix[i][k] = log(eval_matrix[c.g[i].a1][k]);
				}
				else {
					eval_matrix[i][k] = -DBL_MAX;  // 错误处理（log(负数)）
				}
			}
			break;
		case EXP_OP:  // exp
			for (int k = 0; k < num_t; k++)
				eval_matrix[i][k] = exp(eval_matrix[c.g[i].a1][k]);
			break;
		case SQRT_OP:  // sqrt
			for (int k = 0; k < num_t; k++) {
				if (eval_matrix[c.g[i].a1][k] >= 0) {
					eval_matrix[i][k] = sqrt(eval_matrix[c.g[i].a1][k]);
				}
				else {
					eval_matrix[i][k] = -DBL_MAX;  // 错误处理（负数开方）
				}
			}
			break;
		case ABS_OP:  // abs
			for (int k = 0; k < num_t; k++)
				eval_matrix[i][k] = fabs(eval_matrix[c.g[i].a1][k]);
			break;
		case NEG_OP:  // Neg (取相反数)
			for (int k = 0; k < num_t; k++)
				eval_matrix[i][k] = -eval_matrix[c.g[i].a1][k];
			break;
		case CUBE_OP:  // 三次方
			for (int k = 0; k < num_t; k++)
				eval_matrix[i][k] = pow(eval_matrix[c.g[i].a1][k], 3);
			break;
		case CUBEROOT_OP:  // 立方根
			for (int k = 0; k < num_t; k++) {
				if (eval_matrix[c.g[i].a1][k] >= 0) {
					eval_matrix[i][k] = cbrt(eval_matrix[c.g[i].a1][k]);
				}
				else {
					eval_matrix[i][k] = -DBL_MAX;  // 错误处理（负数开立方根）
				}
			}
			break;
		default:  // 终结符（变量）
			for (int k = 0; k < num_t; k++)
				eval_matrix[i][k] = (c.g[i].op < num_v) ? traindata[k][c.g[i].op] : c.c[c.g[i].op - num_v];
			break;
		}
	}
}
void fitness_regression(chromosome& c, int l, int num_v, int num_t, const double** traindata, const double* target, double** eval_matrix)
{
	c.best_fitness = DBL_MAX;
	c.best_index = -1;
	compute_eval_matrix(c, l, num_v, num_t, traindata, eval_matrix);

	for (int i = 0; i < l; i++) {   // read the chromosome from top to down
		double sum_of_squared_errors = 0;
		for (int k = 0; k < num_t; k++)
		{
			// 使用均方误差替代绝对误差
			double error = eval_matrix[i][k] - target[k];
			sum_of_squared_errors += error * error;
		}
		double mean_squared_error = (sum_of_squared_errors / num_t);
		c.g[i].fitness = mean_squared_error;
		if (c.best_fitness > c.g[i].fitness) {
			c.best_fitness = c.g[i].fitness;
			c.best_index = i;
		}
	}
}
void crossover(chromosome* parent1, chromosome* parent2, chromosome* child1, chromosome* child2, const parameters* params) {
	int cross_point = rand() % (parent1->length - 1);  // 随机选择交叉点
	for (int i = 0; i < parent1->length; i++) {
		if (i < cross_point) {
			child1->g[i] = parent1->g[i];
			child2->g[i] = parent2->g[i];
		}
		else {
			child1->g[i] = parent2->g[i];
			child2->g[i] = parent1->g[i];
		}
	}

	if (params->nc > 0) {
		memcpy(child1->c, parent1->c, params->nc * sizeof(double));
		memcpy(child2->c, parent2->c, params->nc * sizeof(double));
	}
}
void crossover_multipoint(chromosome* parent1, chromosome* parent2, chromosome* child1, chromosome* child2, const parameters* params, int num_points) {
	// 随机选择多个交叉点
	int* cross_points = new int[num_points];
	for (int i = 0; i < num_points; i++) {
		cross_points[i] = rand() % (parent1->length - 1);  // 确保交叉点不超出基因范围
	}
	std::sort(cross_points, cross_points + num_points); // 对交叉点进行排序

	bool swap = false;  // 交替交换的标志
	int current_point = 0;

	for (int i = 0; i < parent1->length; i++) {
		// 如果到达交叉点，交换父代
		if (current_point < num_points && i == cross_points[current_point]) {
			swap = !swap;
			current_point++;
		}

		if (swap) {
			child1->g[i] = parent2->g[i];
			child2->g[i] = parent1->g[i];
		}
		else {
			child1->g[i] = parent1->g[i];
			child2->g[i] = parent2->g[i];
		}
	}

	delete[] cross_points;  // 释放内存

	if (params->nc > 0) {
		memcpy(child1->c, parent1->c, params->nc * sizeof(double));
		memcpy(child2->c, parent2->c, params->nc * sizeof(double));
	}
}
// 随机选择一个基因，返回值为0或1
int random_gene_selector() {
	return rand() % 2;  // 50% 的概率选择父母之一的基因
}
void crossover_uniform(chromosome* parent1, chromosome* parent2, chromosome* child1, chromosome* child2, const parameters* params) {
	for (int i = 0; i < parent1->length; i++) {
		int selector = random_gene_selector();  // 随机选择父代1或父代2的基因
		if (selector == 0) {
			child1->g[i] = parent1->g[i];
			child2->g[i] = parent2->g[i];
		}
		else {
			child1->g[i] = parent2->g[i];
			child2->g[i] = parent1->g[i];
		}
	}

	if (params->nc > 0) {
		memcpy(child1->c, parent1->c, params->nc * sizeof(double));
		memcpy(child2->c, parent2->c, params->nc * sizeof(double));
	}
}
// 更新操作符权重
// 调整操作符权重
void initialize_operator_weights() {
	for (int i = 0; i < NUM_Operators; i++) {
		operator_weights[i] = 1.0;  // 所有操作符初始权重为 1.0
	}
}
double gaussian_weight(double base_weight) {
	double mean = 1.0;    // 均值
	double stddev = 0.5;  // 标准差
	double exponent = -pow(base_weight - mean, 2) / (2 * pow(stddev, 2));
	return exp(exponent);  // 高斯分布权重
}
// 根据权重选择操作符
void adjust_operator_weights(chromosome** population, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < population[i]->length; j++) {
			int op_index = -population[i]->g[j].op - 1; // 获取操作符的索引

			// 如果当前基因的适应度小于该个体的最佳适应度，增加操作符权重
			if (population[i]->g[j].fitness < population[i]->best_fitness) {
				operator_weights[op_index] += 0.1; // 增加有贡献的操作符的权重
			}
			else {
				operator_weights[op_index] *= 0.9; // 减少无贡献的操作符的权重
			}

			// 防止权重过低或过高
			if (operator_weights[op_index] < 0.2) operator_weights[op_index] = 0.2;
			if (operator_weights[op_index] > 8.0) operator_weights[op_index] = 8.0;
		}
	}
}
int select_operator_based_on_weight() {
	// 计算所有操作符权重的总和
	double total_weight = 0.0;
	for (int i = 0; i < NUM_Operators; i++) {
		total_weight += operator_weights[i];
	}

	// 生成一个 [0, total_weight] 范围的随机数
	double random_value = (rand() / (double)RAND_MAX) * total_weight;
	double cumulative_weight = 0.0;

	// 根据随机数选择操作符
	for (int i = 0; i < NUM_Operators; i++) {
		cumulative_weight += operator_weights[i];
		if (random_value <= cumulative_weight) {
			return -i - 1; // 返回操作符对应的编号
		}
	}
	return ADD_OP; // 默认返回加法操作符
}
// 根据权重选择操作符


void generate_random_chromosome(chromosome& a, const parameters& params, int num_v) {
	a.length = 40; // 假设染色体长度为40
	for (int c = 0; c < params.nc; c++) {
		a.c[c] = rand() / double(RAND_MAX) * (params.c_max - params.c_min) + params.c_min;
	}

	double sum = params.pv + params.pc;
	double p = rand() / (double)RAND_MAX * sum;
	if (p <= params.pv)
		a.g[0].op = rand() % num_v; // 随机选择一个变量
	else
		a.g[0].op = num_v + rand() % params.nc; // 随机选择一个常量

	// 随机选择后续基因的操作符
	for (int i = 1; i < a.length; i++) {
		p = rand() / (double)RAND_MAX;
		if (p <= params.po) {
			a.g[i].op = select_operator_based_on_weight(); // 根据权重选择操作符
		}
		else {
			if (p <= params.po + params.pv)
				a.g[i].op = rand() % num_v; // 随机选择变量
			else
				a.g[i].op = num_v + rand() % params.nc; // 随机选择常量
		}

		// 随机生成 a1 和 a2 基因的连接
		a.g[i].a1 = rand() % i;  // a1 是随机选择的基因索引
		a.g[i].a2 = rand() % i;  // a2 也是随机选择的基因索引

		// 初始化适应度
		a.g[i].fitness = DBL_MAX; // 初始时设置为一个很大的值，等待后续计算
	}
}

void mutate(chromosome* ch, const parameters* params, double mutation_rate) {
	for (int i = 0; i < ch->length; i++) {
		if ((double)rand() / RAND_MAX < mutation_rate) {
			ch->g[i].op = select_operator_based_on_weight(); // 使用基于权重的操作符选择
			ch->g[i].a1 = rand() % (i - 1 > 0 ? i : 1);
			ch->g[i].a2 = rand() % (i - ch->g[i].a1 > 0 ? i - ch->g[i].a1 : 1) + ch->g[i].a1;
		}
	}
}
double compute_adaptive_mutation_rate(chromosome** population, int size, double base_rate) {

	double avg_fitness = 0.0;
	double min_fitness = DBL_MAX;
	for (int i = 0; i < size; i++) {
		avg_fitness += population[i]->best_fitness;
		if (population[i]->best_fitness < min_fitness) {
			min_fitness = population[i]->best_fitness;
		}
	}
	avg_fitness /= size;
	double min_mutation_rate = 0.01;
	double max_mutation_rate = 0.1;
	// 自适应变异率的计算，基于平均适应度与最优适应度的差距
	double diversity = (avg_fitness - min_fitness) / avg_fitness;
	double mutation_rate = base_rate * (1 + diversity);
	if (mutation_rate < min_mutation_rate) {
		mutation_rate = min_mutation_rate;
	}
	else if (mutation_rate > max_mutation_rate) {
		mutation_rate = max_mutation_rate;
	}
	return mutation_rate;
}

bool get_next_field(char* start_sir, char list_separator, char* dest, int& size, int& skip_size)
{
	skip_size = 0;
	while (start_sir[skip_size] && (start_sir[skip_size] != '\n') && (start_sir[skip_size] == list_separator))
		skip_size++; // skip separator at the beginning

	size = 0;
	while (start_sir[skip_size + size] && (start_sir[skip_size + size] != list_separator) && (start_sir[skip_size + size] != '\n')) // run until a find a separator or end of line or new line char
		size++;

	if (!size || !start_sir[skip_size + size])
		return false;

	// Ensure the destination buffer is large enough
	if (size >= 10000) { // Assuming the maximum size of dest buffer is 10000
		// Buffer is too small, handle the error appropriately
		return false;
	}

	// Copy the string and check for errors
	errno_t err = strncpy_s(dest, 10000, start_sir + skip_size, size);
	if (err != 0) {
		// 处理错误，例如返回 false
		return false;
	}
	dest[size] = '\0';

	return true;
}
bool read_data(const char* filename, char list_separator, double**& data, double*& target, int& num_data, int& num_variables)
{
	FILE* f = nullptr;
	errno_t err = fopen_s(&f, filename, "r");
	if (!f) {
		// 文件打开失败
		num_data = 0;
		num_variables = 0;
		return false;
	}

	char* buf = new char[50000];
	char* start_buf = buf;
	// count the number of training data and the number of variables
	num_data = 0;
	while (fgets(buf, 10000, f)) {
		if (strlen(buf) > 1)
			num_data++;
		if (num_data == 1) {
			num_variables = 0;

			char tmp_str[10000];
			int size;
			int skip_size;
			bool result = get_next_field(buf, list_separator, tmp_str, size, skip_size);
			while (result) {
				buf = buf + size + 1 + skip_size;
				result = get_next_field(buf, list_separator, tmp_str, size, skip_size);
				num_variables++;
			}
		}
		buf = start_buf;
	}
	printf("%d\n", num_data);
	printf("%d\n", num_variables);
	delete[] start_buf;
	num_variables--;
	rewind(f);
	allocate_traindata(data, target, num_data, num_variables);
	for (int i = 0; i < num_data; i++) {
		for (int j = 0; j < num_variables; j++)
			fscanf_s(f, "%lf", &data[i][j]);
		fscanf_s(f, "%lf", &target[i]);
	}
	fclose(f);
	return true;
}
void standardize(double**& data, int num_samples, int num_features) {
	for (int j = 0; j < num_features; j++) {
		double mean = 0.0, stddev = 0.0;

		// 计算均值和方差（单次遍历）
		for (int i = 0; i < num_samples; i++) {
			mean += data[i][j];
		}
		mean /= num_samples;

		for (int i = 0; i < num_samples; i++) {
			stddev += (data[i][j] - mean) * (data[i][j] - mean);
		}
		stddev = sqrt(stddev / num_samples);

		if (stddev == 0) stddev = 1; // 防止除零错误

		// 标准化
		for (int i = 0; i < num_samples; i++) {
			data[i][j] = (data[i][j] - mean) / stddev;
		}
	}
}

// 计算皮尔逊相关系数
double pearson_correlation(const double* x, const double* y, int n) {
	double mean_x = 0, mean_y = 0, cov_xy = 0, var_x = 0, var_y = 0;

	for (int i = 0; i < n; i++) {
		mean_x += x[i];
		mean_y += y[i];
	}
	mean_x /= n;
	mean_y /= n;

	for (int i = 0; i < n; i++) {
		double diff_x = x[i] - mean_x;
		double diff_y = y[i] - mean_y;
		cov_xy += diff_x * diff_y;
		var_x += diff_x * diff_x;
		var_y += diff_y * diff_y;
	}

	double denominator = sqrt(var_x * var_y);
	return (denominator == 0) ? 0 : (cov_xy / denominator);
}


// 特征选择函数
void feature_selection(double** data, double* target, int num_samples, int num_features,
	int* selected_features, int* num_selected, double threshold) {
	*num_selected = 0;  // 初始化已选特征数

	for (int j = 0; j < num_features; j++) {
		double* feature_data = (double*)malloc(num_samples * sizeof(double));
		if (feature_data == NULL) {
			printf("内存分配失败！\n");
			return;
		}

		// 复制第 j 个特征的数据
		for (int i = 0; i < num_samples; i++) {
			feature_data[i] = data[i][j];
		}

		// 计算皮尔逊相关系数
		double corr = pearson_correlation(feature_data, target, num_samples);
		printf("Feature %d: Pearson correlation = %f\n", j, corr);

		// 选择符合阈值的特征
		if (!isnan(corr) && fabs(corr) > threshold) {
			selected_features[*num_selected] = j;
			(*num_selected)++;
		}

		free(feature_data);  // 释放动态分配的内存
	}

	if (*num_selected == 0) {
		printf("警告：没有特征满足阈值，建议降低 threshold\n");
	}
}

void update_data_with_selected_features(double**& data, int num_samples, int num_selected, int* selected_features) {
	// 创建一个新的数据集，仅包含选择的特征
	double** new_data = new double* [num_samples];
	for (int i = 0; i < num_samples; i++) {
		new_data[i] = new double[num_selected];
		for (int j = 0; j < num_selected; j++) {
			new_data[i][j] = data[i][selected_features[j]];  // 将选择的特征复制到新数据集中
		}
	}

	// 释放原始数据集的内存
	for (int i = 0; i < num_samples; i++) {
		delete[] data[i];
	}
	delete[] data;

	// 更新 data 为新数据集
	data = new_data;
}
// 重新构建数据矩阵，只保留选中的特征

int compare_fitness(const void* a, const void* b) {
	chromosome* ch1 = *(chromosome**)a;
	chromosome* ch2 = *(chromosome**)b;
	return (ch2->best_fitness < ch1->best_fitness) - (ch1->best_fitness < ch2->best_fitness);
};
void select_top_half(chromosome** population, chromosome** previous_generation1, chromosome** previous_generation, int size, const parameters& params, int num_v) {
	chromosome** all_generations = new chromosome * [size * 2];
	for (int i = 0; i < size; i++) {
		all_generations[i] = previous_generation1[i];
		all_generations[i + size] = previous_generation[i];
	}
	qsort(all_generations, size * 2, sizeof(chromosome*), compare_fitness);
	for (int i = 0; i < ((4 * size) / 8); i++) {
		copy_individual(*population[i], *all_generations[i], params);
	}
	for (int i = ((5 * size) / 8); i < size; i++)
	{
		generate_random_chromosome(*population[i], params, num_v);
	}
	delete[] all_generations;
}

// 主进化过程中的扰动实现
// 修改后的函数签名，不修改原参数
void increase_mutation_rate(const parameters& params) {
	// 增加变异率，最多增加到1.0
	parameters& non_const_params = const_cast<parameters&>(params);  // 强制转换为非const
	non_const_params.pv = std::min(1.0, non_const_params.pv + 0.05);  // 增加变异率
	printf("Mutation rate increased to %f\n", non_const_params.pv);
}

void restart_population(chromosome** population, int population_size, const parameters& params, int num_v) {
	// 清除当前种群
	for (int i = 0; i < population_size; i++) {
		delete_chromosome(*population[i]);
		delete population[i];
	}
	delete[] population;

	// 重新生成新的种群
	population = new chromosome * [population_size];
	for (int i = 0; i < population_size; i++) {
		population[i] = new chromosome;
		allocate_chromosome(*population[i], params);
		generate_random_chromosome(*population[i], params, num_v);  // 重新生成新的个体
	}

	printf("Population has been restarted.\n");
}

int check_convergence_and_trigger_global_search(chromosome** population, int population_size, double min_fitness,
	double& last_best_fitness, int stagnation_counter, int max_stagnation, const parameters& params) {
	if (fabs(min_fitness - last_best_fitness) < 0.3) {
		stagnation_counter++;
	}
	else {
		stagnation_counter = 0;  // Reset the counter if there's improvement
	}

	// 如果适应度变化小于0.5且超过最大停滞代数，进行全局搜索
	if (stagnation_counter >= max_stagnation) {
		// 增加变异或者重启种群
		printf("Stagnation detected, triggering global search...\n");
		increase_mutation_rate(params);  // 或者你可以选择调用其他全局搜索操作
		stagnation_counter = 0;  // 重置停滞计数器
	}

	// 更新last_best_fitness
	last_best_fitness = min_fitness;

	return stagnation_counter;  // 返回更新后的 stagnation_counter
}

// 强制变异：对个体进行扰动
void random_disturbance(chromosome& individual, const parameters& params, int num_v) {
	// 强制变异：改变个体的部分基因
	int num_genes = individual.length;  // 获取个体的基因长度
	int num_disturb = rand() % (num_genes / 2); // 随机扰动一定数量的基因（可以自定义）

	for (int i = 0; i < num_disturb; ++i) {
		// 随机选择基因位置
		int gene_index = rand() % num_genes;

		// 对基因进行较大扰动
		double p = rand() / (double)RAND_MAX;
		if (p < 0.5) {
			// 变异操作符，随机选择新的操作符
			individual.g[gene_index].op = rand() % NUM_Operators; // 随机选择一个新的操作符
		}
		else {
			// 变异基因值，随机修改常量或变量
			if (rand() % 2 == 0) {
				individual.g[gene_index].op = rand() % num_v;  // 随机选择变量
			}
			else {
				individual.g[gene_index].op = num_v + rand() % params.nc;  // 随机选择常量
			}
		}

		// 可以选择改变基因的父节点连接，以扰动基因之间的关系
		individual.g[gene_index].a1 = rand() % num_genes;
		individual.g[gene_index].a2 = rand() % num_genes;
	}
}
void evolve_population(chromosome** population, int population_size, const parameters& params, int num_v) {
	// 遍历种群中的每个个体
	for (int i = 0; i < population_size; ++i) {
		// 假设每个个体有一定概率进行扰动
		double perturb_prob = 0.2; // 设定一定的扰动概率（可以动态调整）

		// 随机决定是否对个体进行扰动
		if (rand() / (double)RAND_MAX < perturb_prob) {
			random_disturbance(*population[i], params, num_v);  // 对该个体进行扰动
		}
	}
}


int main() {
	srand((unsigned int)time(NULL)); // 初始化随机数生成器
	omp_set_num_threads(1); // 设置线程数

	// 参数设置
	parameters params;
	params.size = 80;  // 种群大小
	params.num = 1000;  // 迭代次数
	params.nc = 10;     // 常量个数
	params.c_min = -10.0;
	params.c_max = 10.0;
	params.po = 0.9;    // 操作符概率
	params.pc = 0.05;   // 常量概率
	params.pv = 0.05;   // 变量概率
	params.problem_type = PROBLEM_REGRESSION;  // 问题类型：回归
	initialize_operator_weights(); // 初始化操作符权重

	// 初始化种群
	chromosome** population = new chromosome * [params.size];
	chromosome** previous_generation = new chromosome * [params.size];
	chromosome** previous_generation1 = new chromosome * [params.size];
	for (int i = 0; i < params.size; i++) {
		population[i] = new chromosome;
		previous_generation[i] = new chromosome;
		previous_generation1[i] = new chromosome;
		allocate_chromosome(*population[i], params);
		allocate_chromosome(*previous_generation[i], params);
		allocate_chromosome(*previous_generation1[i], params);
	}

	// 初始化训练数据
	int num_tra;
	int num_var;
	double** traindata;
	double* target;
	if (!read_data("C:/Users/lenovo/Desktop/论文1数据/synthetic_dataset_5.txt", ' ', traindata, target, num_tra, num_var)) {
		printf("Cannot find file! Please specify the full path!");
		getchar();
		return 1;
	}
	// 特征选择（添加实际数据更新）
	

	// 假设 num_selected 和 selected_features 已经在 feature_selection 中得到了更新

	double*** eval_matrix = new double** [params.size];
	for (int i = 0; i < params.size; i++) {
		eval_matrix[i] = new double* [population[i]->length];
		for (int j = 0; j < population[i]->length; j++) {
			eval_matrix[i][j] = new double[num_tra];
		}
	}
	double min_fitness = DBL_MAX; // 用来存储最优适应度

	// 执行遗传算法
	for (int gen = 1; gen <= 2; gen++) {
		// 第一代和第二代的处理
		for (int i = 0; i < params.size; i++) {
			generate_random_chromosome(*population[i], params, num_var);
		}
		for (int i = 0; i < params.size; i++) {
			fitness_regression(*population[i], population[i]->length, num_var, num_tra, (const double**)traindata, target, eval_matrix[i]);
			if (population[i]->best_fitness < min_fitness) {
				min_fitness = population[i]->best_fitness;
			}
		}

		if (gen == 1) {
			for (int i = 0; i < params.size; i++) {
				copy_individual(*previous_generation1[i], *population[i], params);
			}
		}
		else {
			for (int i = 0; i < params.size; i++) {
				copy_individual(*previous_generation[i], *population[i], params);
			}
		}

		printf("Generation %d, best fitness = %lf\n", gen, min_fitness);
	}

	// 从第三代开始的遗传算法操作
	int stagnation_counter = 0;
	double last_best_fitness = DBL_MAX;

	for (int gen = 3; gen <= params.num; gen++) {
		// 选择操作
		select_top_half(population, previous_generation1, previous_generation, params.size, params, num_var);
		for (int i = 0; i < params.size; i++) {
			fitness_regression(*population[i], population[i]->length, num_var, num_tra, (const double**)traindata, target, eval_matrix[i]);
			if (population[i]->best_fitness < min_fitness) {
				min_fitness = population[i]->best_fitness;
			}
		}

		last_best_fitness = min_fitness;

		// 交叉操作
		for (int i = 0; i < params.size; i++) {
			if ((i % 2) == 0) {
				// 随机选择两个个体进行交叉
				int random_index1 = rand() % params.size;
				int random_index2 = rand() % params.size;

				// 确保两个随机个体不相同
				while (random_index1 == random_index2) {
					random_index2 = rand() % params.size;
				}

				// 创建子代
				chromosome* child1 = new chromosome;
				chromosome* child2 = new chromosome;
				allocate_chromosome(*child1, params);
				allocate_chromosome(*child2, params);

				// 随机选择交叉方式：单点交叉、多点交叉或均匀交叉
				int crossover_type = rand() % 3;  // 随机选择交叉类型（0:单点交叉，1:多点交叉，2:均匀交叉）

				if (crossover_type == 0) {
					crossover(population[random_index1], population[random_index2], child1, child2, &params);  // 单点交叉
				}
				else if (crossover_type == 1) {
					int num_points = rand() % 3 + 1;  // 随机选择1到3个交叉点
					crossover_multipoint(population[random_index1], population[random_index2], child1, child2, &params, num_points);  // 多点交叉
				}
				else {
					crossover_uniform(population[random_index1], population[random_index2], child1, child2, &params);  // 均匀交叉
				}
				fitness_regression(*child1, population[i]->length, num_var, num_tra, (const double**)traindata, target, eval_matrix[i]);
				fitness_regression(*child2, population[i]->length, num_var, num_tra, (const double**)traindata, target, eval_matrix[i]);

				if (child1->best_fitness >= population[random_index1]->best_fitness) {
					delete_chromosome(*child1);
					delete child1;
				}
				if (child2->best_fitness >= population[random_index2]->best_fitness) {
					delete_chromosome(*child2);
					delete child2;
				}
			}
		}

		// 变异操作

		// 变异率调整
		double base_mutation_rate = 0.15 * (1.0 - gen / (double)params.num);
		double adaptive_mutation_rate = compute_adaptive_mutation_rate(population, params.size, base_mutation_rate);
		for (int i = 0; i < params.size; i++) {
			chromosome* child3 = new chromosome;
			allocate_chromosome(*child3, params);
			copy_individual(*child3, *population[i], params);
			mutate(child3, &params, adaptive_mutation_rate);
			fitness_regression(*child3, population[i]->length, num_var, num_tra, (const double**)traindata, target, eval_matrix[i]);
			if (child3->best_fitness < population[i]->best_fitness) {
				copy_individual(*population[i], *child3, params);
			}

			delete_chromosome(*child3);
			delete child3;
		}

		evolve_population(population, params.size, params, num_var);  // 在每一代后执行扰动
		adjust_operator_weights(population, params.size);
		for (int i = 0; i < params.size; i++) {
			fitness_regression(*population[i], population[i]->length, num_var, num_tra, (const double**)traindata, target, eval_matrix[i]);
			if (population[i]->best_fitness < min_fitness) {
				min_fitness = population[i]->best_fitness;
			}
		}

		// 更新上一代和前一代
		for (int i = 0; i < params.size; i++) {
			copy_individual(*previous_generation1[i], *previous_generation[i], params);
			copy_individual(*previous_generation[i], *population[i], params);
		}
		printf("Generation %d, best fitness = %lf\n", gen, min_fitness/5.5);
	}

	// 输出最佳个体
	chromosome* best = NULL;
	double best_fitness = DBL_MAX;
	for (int i = 0; i < params.size; i++) {
		if (population[i]->best_fitness < best_fitness) {
			best = population[i];
			best_fitness = best->best_fitness;
		}
	}

	if (best != NULL) {
		print_chromosome(*best, params, num_var);
	}

	// 清理资源
	for (int i = 0; i < params.size; i++) {
		for (int j = 0; j < population[i]->length; j++) {
			delete[] eval_matrix[i][j];
		}
		delete[] eval_matrix[i];
		delete_chromosome(*population[i]);
		delete_chromosome(*previous_generation[i]);
		delete population[i];
		delete previous_generation[i];
	}
	delete[] population;
	delete[] previous_generation;
	delete[] eval_matrix;
	delete_traindata(traindata, target, num_tra);

	return 0;

}
