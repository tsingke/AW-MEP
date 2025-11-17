//---------------------------------------------------------------------------
//	Multi Expression Programming
//---------------------------------------------------------------------------

//   More info at:  
//     https://mepx.org
//     https://mepx.github.io
//     https://github.com/mepx

//   Compiled with Microsoft Visual C++ 2019
//   Also compiled with XCode 9.

//   Please reports any sugestions and/or bugs to mihai.oltean@gmail.com

//   Training data file must have the following format (see building1.txt and cancer1.txt from the dataset folder):
//   Note that building1 and cancer1 data were taken from PROBEN1

//   x_11 x_12 ... x_1n f_1
//   x_21 x_22 ....x_2n f_2
//   .............
//   x_m1 x_m2 ... x_mn f_m

//   where m is the number of training data
//   and n is the number of variables.
//   x_ij are the inputs
//   f_i are the outputs

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <float.h> 
#define PROBLEM_REGRESSION 0
#define PROBLEM_BINARY_CLASSIFICATION 1

#define NUM_Operators 18//定义操作符数量，有5个操作符

#define ADD_OP -1 // +
#define DIF_OP -2 // -
#define MUL_OP -3 // *
#define DIV_OP -4 // /
#define SIN_OP -5 // s
#define MOD_OP -6  // %
#define POW_OP -7  // ^
#define AND_OP -8  // &
#define OR_OP -9   // |
#define TAN_OP -10    // 正切函数
#define LOG_OP -11   // 自然对数
#define EXP_OP -12   // 指数函数
#define SQRT_OP -13
#define ABS_OP -14
#define FLOOR_OP -15 // 向下取整
#define CEIL_OP -16  // 向上取整
#define MIN_OP -17
#define MAX_OP -18
char operators_string[NUM_Operators + 1] = "+-*/s%^&|tleLAFCM";

//---------------------------------------------------------------------------
struct t_code3 {
	int op;				// either a variable, an operator or a constant
	// variables are indexed from 0: 0, 1, 2, ...
	// constants are indexed from num_variables; the first constant has index num_variables
	// operators are stored as negative numbers -1, -2, -3...
	int addr1, addr2;    // pointers to arguments
};
//定义结构体，基因---------------------------------------------------------------------------
struct t_mep_chromosome {
	t_code3* code;        // the program - a string of genes
	double* constants;   // an array of constants

	double fitness;        // the fitness (or the error)
	// for regression is computed as sum of abs differences between target and obtained
	// for binary classification is computed as the number of incorrectly classified data
	int best_index;        // the index of the best expression in chromosome
};
//定义染色体---------------------------------------------------------------------------
struct t_mep_parameters {
	int code_length;             // number of instructions in a chromosome
	int num_generations;
	int pop_size;                // population size
	double mutation_probability, crossover_probability;
	int num_constants;
	double constants_min, constants_max;   // the array for constants
	double variables_probability, operators_probability, constants_probability;

	int problem_type; //0 - regression, 1 - classification
	double classification_threshold; // for classification problems only
};
//参数配置---------------------------------------------------------------------------
void allocate_chromosome(t_mep_chromosome& c, const t_mep_parameters& params)
{
	c.code = new t_code3[params.code_length];// the code
	if (params.num_constants)
		c.constants = new double[params.num_constants];// constants
	else
		c.constants = NULL;
}
//---------------------------------------------------------------------------
void allocate_chromosomeline(t_mep_chromosome& c, const t_mep_parameters& params, int t)
{
	c.code = new t_code3[(params.code_length - (t + 1))];// the code
	if (params.num_constants)
		c.constants = new double[params.num_constants];// constants
	else
		c.constants = NULL;
}
void delete_chromosome(t_mep_chromosome& c)
{
	if (c.code) {
		delete[] c.code;
		c.code = NULL;
	}
	if (c.constants) {
		delete[] c.constants;
		c.constants = NULL;
	}
}
//---------------------------------------------------------------------------
void allocate_training_data(double**& data, double*& target, int num_training_data, int num_variables)
{
	target = new double[num_training_data];
	data = new double* [num_training_data];
	for (int i = 0; i < num_training_data; i++)
		data[i] = new double[num_variables];
}
//---------------------------------------------------------------------------
void allocate_partial_expression_values(double**& expression_value, int num_training_data, int code_length)
{// allocate memory for the matrix storing the output of each expression for each training data
	// this is allocated once and then reused, in order to reduce the number of allocations/deletions
	expression_value = new double* [code_length];
	for (int i = 0; i < code_length; i++)
		expression_value[i] = new double[num_training_data];
}
//---------------------------------------------------------------------------
void delete_partial_expression_values(double**& expression_value, int code_length)
{
	if (expression_value) {
		for (int i = 0; i < code_length; i++)
			delete[] expression_value[i];
		delete[] expression_value;
	}
}
//---------------------------------------------------------------------------
void delete_training_data(double**& data, double*& target, int num_training_data)
{
	if (data)
		for (int i = 0; i < num_training_data; i++)
			delete[] data[i];
	delete[] data;
	delete[] target;
}
//---------------------------------------------------------------------------
void copy_individual(t_mep_chromosome& dest, const t_mep_chromosome& source, const t_mep_parameters& params)
{
	for (int i = 0; i < params.code_length; i++)
		dest.code[i] = source.code[i];
	for (int i = 0; i < params.num_constants; i++)
		dest.constants[i] = source.constants[i];
	dest.fitness = source.fitness;
	dest.best_index = source.best_index;
}
//---------------------------------------------------------------------------
void generate_random_chromosome(t_mep_chromosome& a, const t_mep_parameters& params, int num_variables)
// randomly initializes the individuals
{
	// generate constants first
	for (int c = 0; c < params.num_constants; c++)
		a.constants[c] = rand() / double(RAND_MAX) * (params.constants_max - params.constants_min) + params.constants_min;

	// on the first position we can have only a variable or a constant
	double sum = params.variables_probability + params.constants_probability;
	double p = rand() / (double)RAND_MAX * sum;

	if (p <= params.variables_probability)
		a.code[0].op = rand() % num_variables;
	else
		a.code[0].op = num_variables + rand() % params.num_constants;

	// for all other genes we put either an operator, variable or constant
	for (int i = 1; i < params.code_length; i++) {
		p = rand() / (double)RAND_MAX;

		if (p <= params.operators_probability)
			a.code[i].op = -rand() % NUM_Operators - 1;        // an operator
		else {
			if (p <= params.operators_probability + params.variables_probability)
				a.code[i].op = rand() % num_variables;     // a variable
			else
				a.code[i].op = num_variables + rand() % params.num_constants; // index of a constant
		}
		a.code[i].addr1 = rand() % i;
		a.code[i].addr2 = rand() % i;
	}
}
//---------------------------------------------------------------------------
void generate_random_chromosomeline(t_mep_chromosome& a, const t_mep_parameters& params, int num_variables, int t)
// randomly initializes the individuals
{
	// generate constants first
	for (int c = 0; c < params.num_constants; c++)
		a.constants[c] = rand() / double(RAND_MAX) * (params.constants_max - params.constants_min) + params.constants_min;

	// on the first position we can have only a variable or a constant
	double sum = params.variables_probability + params.constants_probability;
	double p = rand() / (double)RAND_MAX * sum;
	// for all other genes we put either an operator, variable or constant
	if (p <= params.variables_probability)
		a.code[0].op = rand() % num_variables;
	else
		a.code[0].op = num_variables + rand() % params.num_constants;
	int m = t + 1;
	for (int i = 1; i < (params.code_length - m); i++) {
		p = rand() / (double)RAND_MAX;

		if (p <= params.operators_probability)
			a.code[i].op = -rand() % NUM_Operators - 1;        // an operator
		else {
			if (p <= params.operators_probability + params.variables_probability)
				a.code[i].op = rand() % num_variables;     // a variable
			else
				a.code[i].op = num_variables + rand() % params.num_constants; // index of a constant
		}
		a.code[i].addr1 = rand() % t;
		a.code[i].addr2 = rand() % t;
	}
}
void compute_eval_matrix(const t_mep_chromosome& c, int code_length, int num_variables,
	int num_training_data, const double** training_data,
	double** eval_matrix)
{
	// we keep intermediate values in a matrix because when an error occurs (like division by 0) we mutate that gene into a variables.
	// in such case it is faster to have all intermediate results until current gene, so that we don't have to recompute them again.

	for (int i = 0; i < code_length; i++) {   // read the chromosome from top to down
		bool is_error_case = false;// division by zero, other errors
		switch (c.code[i].op) {

		case  ADD_OP:  // +
			for (int k = 0; k < num_training_data; k++)
				eval_matrix[i][k] = eval_matrix[c.code[i].addr1][k] + eval_matrix[c.code[i].addr2][k];
			break;
		case  DIF_OP:  // -
			for (int k = 0; k < num_training_data; k++)
				eval_matrix[i][k] = eval_matrix[c.code[i].addr1][k] - eval_matrix[c.code[i].addr2][k];

			break;
		case  MUL_OP:  // *
			for (int k = 0; k < num_training_data; k++)
				eval_matrix[i][k] = eval_matrix[c.code[i].addr1][k] * eval_matrix[c.code[i].addr2][k];
			break;
		case  DIV_OP:  //  /
			for (int k = 0; k < num_training_data; k++)
				if (fabs(eval_matrix[c.code[i].addr2][k]) < 1e-6) // test if it is too small
					is_error_case = true;
			if (is_error_case) {                                           // an division by zero error occured !!!
				c.code[i].op = rand() % num_variables;   // the gene is mutated into a terminal
				for (int k = 0; k < num_training_data; k++)
					eval_matrix[i][k] = training_data[k][c.code[i].op];
			}
			else    // normal execution....
				for (int k = 0; k < num_training_data; k++)
					eval_matrix[i][k] = eval_matrix[c.code[i].addr1][k] / eval_matrix[c.code[i].addr2][k];
			break;
		case  SIN_OP:  // sin
			for (int k = 0; k < num_training_data; k++)
				eval_matrix[i][k] = sin(eval_matrix[c.code[i].addr1][k]);
			break;
		case MOD_OP:  // %
			for (int k = 0; k < num_training_data; k++) {
				if (eval_matrix[c.code[i].addr2][k] == 0) {
					// 处理取模运算中的除零错误
					is_error_case = true;
					break;
				}
				eval_matrix[i][k] = fmod(eval_matrix[c.code[i].addr1][k], eval_matrix[c.code[i].addr2][k]);
			}
			break;
		case POW_OP:  // ^
			for (int k = 0; k < num_training_data; k++) {
				eval_matrix[i][k] = pow(eval_matrix[c.code[i].addr1][k], eval_matrix[c.code[i].addr2][k]);
			}
			break;
		case AND_OP:  // &
			for (int k = 0; k < num_training_data; k++) {
				eval_matrix[i][k] = (int)eval_matrix[c.code[i].addr1][k] & (int)eval_matrix[c.code[i].addr2][k];
			}
			break;
		case OR_OP:   // |
			for (int k = 0; k < num_training_data; k++) {
				eval_matrix[i][k] = (int)eval_matrix[c.code[i].addr1][k] | (int)eval_matrix[c.code[i].addr2][k];
			}
			break;
		case TAN_OP:  // tan
			for (int k = 0; k < num_training_data; k++) {
				eval_matrix[i][k] = tan(eval_matrix[c.code[i].addr1][k]);
			}
			break;
		case LOG_OP:  // log
			for (int k = 0; k < num_training_data; k++) {
				if (eval_matrix[c.code[i].addr1][k] > 0) {
					eval_matrix[i][k] = log(eval_matrix[c.code[i].addr1][k]);
				}
				else {
					// 处理错误情况，例如对数函数的输入必须大于0
					eval_matrix[i][k] = -DBL_MAX; // 使用 -DBL_MAX 或其他方式来表示错误
				}
			}
			break;
		case EXP_OP:  // exp
			for (int k = 0; k < num_training_data; k++) {
				eval_matrix[i][k] = exp(eval_matrix[c.code[i].addr1][k]);
			}
			break;
		case SQRT_OP:  // sqrt
			for (int k = 0; k < num_training_data; k++) {
				if (eval_matrix[c.code[i].addr1][k] >= 0) {
					eval_matrix[i][k] = sqrt(eval_matrix[c.code[i].addr1][k]);
				}
				else {
					// 平方根函数的输入不能是负数
					eval_matrix[i][k] = -DBL_MAX; // 使用 -DBL_MAX 或其他方式来表示错误
				}
			}
			break;
		case ABS_OP:  // abs
			for (int k = 0; k < num_training_data; k++) {
				eval_matrix[i][k] = fabs(eval_matrix[c.code[i].addr1][k]);
			}
			break;
		case FLOOR_OP:  // floor
			for (int k = 0; k < num_training_data; k++) {
				eval_matrix[i][k] = floor(eval_matrix[c.code[i].addr1][k]);
			}
			break;
		case CEIL_OP:  // ceil
			for (int k = 0; k < num_training_data; k++) {
				eval_matrix[i][k] = ceil(eval_matrix[c.code[i].addr1][k]);
			}
			break;
		case MIN_OP:  // fmin
			for (int k = 0; k < num_training_data; k++) {
				eval_matrix[i][k] = fmin(eval_matrix[c.code[i].addr1][k], eval_matrix[c.code[i].addr2][k]);
			}
			break;
		case MAX_OP:  // fmax
			for (int k = 0; k < num_training_data; k++) {
				eval_matrix[i][k] = fmax(eval_matrix[c.code[i].addr1][k], eval_matrix[c.code[i].addr2][k]);
			}
			break;
		default:  // a variable
			for (int k = 0; k < num_training_data; k++)
				if (c.code[i].op < num_variables)
					eval_matrix[i][k] = training_data[k][c.code[i].op];
				else
					eval_matrix[i][k] = c.constants[c.code[i].op - num_variables];
			break;
		}
	}
}
//---------------------------------------------------------------------------
void compute_eval_matrixline(const t_mep_chromosome& d, const t_mep_chromosome& c, int code_length, int num_variables,
	int num_training_data, const double** training_data,
	double** eval_matrix, int t)
{

	for (int i = 0; i < code_length; i++) {   // read the chromosome from top to down
		bool is_error_case = false;// division by zero, other errors
		switch (c.code[i].op) {

		case  ADD_OP:  // +
			for (int k = 0; k < num_training_data; k++)
				eval_matrix[i][k] = eval_matrix[c.code[i].addr1][k] + eval_matrix[c.code[i].addr2][k];
			break;
		case  DIF_OP:  // -
			for (int k = 0; k < num_training_data; k++)
				eval_matrix[i][k] = eval_matrix[c.code[i].addr1][k] - eval_matrix[c.code[i].addr2][k];

			break;
		case  MUL_OP:  // *
			for (int k = 0; k < num_training_data; k++)
				eval_matrix[i][k] = eval_matrix[c.code[i].addr1][k] * eval_matrix[c.code[i].addr2][k];
			break;
		case  DIV_OP:  //  /
			for (int k = 0; k < num_training_data; k++)
				if (fabs(eval_matrix[c.code[i].addr2][k]) < 1e-6) // test if it is too small
					is_error_case = true;
			if (is_error_case) {                                           // an division by zero error occured !!!
				c.code[i].op = rand() % num_variables;   // the gene is mutated into a terminal
				for (int k = 0; k < num_training_data; k++)
					eval_matrix[i][k] = training_data[k][c.code[i].op];
			}
			else    // normal execution....
				for (int k = 0; k < num_training_data; k++)
					eval_matrix[i][k] = eval_matrix[c.code[i].addr1][k] / eval_matrix[c.code[i].addr2][k];
			break;
		case  SIN_OP:  // sin
			for (int k = 0; k < num_training_data; k++)
				eval_matrix[i][k] = sin(eval_matrix[c.code[i].addr1][k]);
			break;
		case MOD_OP:  // %
			for (int k = 0; k < num_training_data; k++) {
				if (eval_matrix[c.code[i].addr2][k] == 0) {
					// 处理取模运算中的除零错误
					is_error_case = true;
					break;
				}
				eval_matrix[i][k] = fmod(eval_matrix[c.code[i].addr1][k], eval_matrix[c.code[i].addr2][k]);
			}
			break;
		case POW_OP:  // ^
			for (int k = 0; k < num_training_data; k++) {
				eval_matrix[i][k] = pow(eval_matrix[c.code[i].addr1][k], eval_matrix[c.code[i].addr2][k]);
			}
			break;
		case AND_OP:  // &
			for (int k = 0; k < num_training_data; k++) {
				eval_matrix[i][k] = (int)eval_matrix[c.code[i].addr1][k] & (int)eval_matrix[c.code[i].addr2][k];
			}
			break;
		case OR_OP:   // |
			for (int k = 0; k < num_training_data; k++) {
				eval_matrix[i][k] = (int)eval_matrix[c.code[i].addr1][k] | (int)eval_matrix[c.code[i].addr2][k];
			}
			break;
		default:  // a variable
			for (int k = 0; k < num_training_data; k++)
				if (c.code[i].op < num_variables)
					eval_matrix[i][k] = training_data[k][c.code[i].op];
				else
					eval_matrix[i][k] = c.constants[c.code[i].op - num_variables];
			break;
		}
	}
}
void fitness_regression(t_mep_chromosome& c, int code_length, int num_variables,
	int num_training_data, const double** training_data, const double* target,
	double** eval_matrix)
{
	c.fitness = 1e+308;
	c.best_index = -1;

	compute_eval_matrix(c, code_length, num_variables, num_training_data, training_data, eval_matrix);

	for (int i = 0; i < code_length; i++) {
		double sum_of_squared_errors = 0;
		for (int k = 0; k < num_training_data; k++) {
			double error = eval_matrix[i][k] - target[k];
			sum_of_squared_errors += error * error;
		}
		double mean_squared_error = sum_of_squared_errors / num_training_data;

		if (c.fitness > mean_squared_error) {
			c.fitness = mean_squared_error;
			c.best_index = i;
		}
	}

}
void fitness_regressionline(t_mep_chromosome& d, t_mep_chromosome& c, int code_length, int num_variables,
	int num_training_data, const double** training_data, const double* target,
	double** eval_matrix, int t)
{
	d.fitness = 1e+308;
	d.best_index = -1;

	compute_eval_matrixline(d, c, code_length, num_variables, num_training_data, training_data, eval_matrix, t);

	for (int i = 0; i < code_length; i++) {   // read the chromosome from top to down
		double sum_of_errors = 0;
		for (int k = 0; k < num_training_data; k++) {
			sum_of_errors += fabs(eval_matrix[i][k] - target[k]);// difference between obtained and expected
		}
		if (d.fitness > sum_of_errors) {
			d.fitness = sum_of_errors;
			d.best_index = i;
		}
	}
}
//---------------------------------------------------------------------------
void fitness_classification(t_mep_chromosome& c, const t_mep_parameters& params, int num_variables,
	int num_training_data, const double** training_data, const double* target,
	double** eval_matrix)
{
	// 使用一个阈值
	// 如果小于阈值，则属于类0
	// 否则属于类1

	c.fitness = 1e+308;  // 设置为一个非常大的初始值
	c.best_index = -1;

	compute_eval_matrix(c, params.code_length, num_variables, num_training_data, training_data, eval_matrix); // 计算每个表达式的输出

	for (int i = 0; i < params.code_length; i++) {   // 从上到下读取染色体
		double sum_of_squared_errors = 0;
		for (int k = 0; k < num_training_data; k++)
		{
			// 使用均方误差替代绝对误差
			double error = eval_matrix[i][k] - target[k];
			sum_of_squared_errors += error * error;
		}
		double mean_squared_error = sum_of_squared_errors / num_training_data;  // 计算均方误差

		if (c.fitness > mean_squared_error) {  // 如果当前表达式的均方误差小于当前最优适应度
			c.fitness = mean_squared_error;   // 更新适应度为当前表达式的均方误差
			c.best_index = i;                 // 更新最优表达式的索引
		}
	}
}

//---------------------------------------------------------------------------
void mutation(t_mep_chromosome& a_chromosome, const t_mep_parameters& params, int num_variables)
// mutate the individual
{
	// mutate each symbol with the given probability
	// first gene must be a variable or constant
	double p = rand() / (double)RAND_MAX;
	if (p < params.mutation_probability) {
		double sum = params.variables_probability + params.constants_probability;
		p = rand() / (double)RAND_MAX * sum;

		if (p <= params.variables_probability)
			a_chromosome.code[0].op = rand() % num_variables;
		else
			a_chromosome.code[0].op = num_variables + rand() % params.num_constants;
	}
	// other genes
	for (int i = 1; i < params.code_length; i++) {
		p = rand() / (double)RAND_MAX;      // mutate the operator
		if (p < params.mutation_probability) {
			// we mutate it, but we have to decide what we put here
			p = rand() / (double)RAND_MAX;

			if (p <= params.operators_probability)
				a_chromosome.code[i].op = -rand() % NUM_Operators - 1;
			else {
				if (p <= params.operators_probability + params.variables_probability)
					a_chromosome.code[i].op = rand() % num_variables;
				else
					a_chromosome.code[i].op = num_variables + rand() % params.num_constants; // index of a constant
			}
		}

		p = rand() / (double)RAND_MAX;      // mutate the first address  (addr1)
		if (p < params.mutation_probability)
			a_chromosome.code[i].addr1 = rand() % i;

		p = rand() / (double)RAND_MAX;      // mutate the second address   (addr2)
		if (p < params.mutation_probability)
			a_chromosome.code[i].addr2 = rand() % i;
	}
	// mutate the constants
	for (int c = 0; c < params.num_constants; c++) {
		p = rand() / (double)RAND_MAX;
		if (p < params.mutation_probability)
			a_chromosome.constants[c] = rand() / double(RAND_MAX) * (params.constants_max - params.constants_min) + params.constants_min;
	}

}
//---------------------------------------------------------------------------
void one_cut_point_crossover(const t_mep_chromosome& parent1, const t_mep_chromosome& parent2,
	const t_mep_parameters& params,
	t_mep_chromosome& offspring1, t_mep_chromosome& offspring2)
{
	int cutting_pct = rand() % params.code_length;
	for (int i = 0; i < cutting_pct; i++) {
		offspring1.code[i] = parent1.code[i];
		offspring2.code[i] = parent2.code[i];
	}
	for (int i = cutting_pct; i < params.code_length; i++) {
		offspring1.code[i] = parent2.code[i];
		offspring2.code[i] = parent1.code[i];
	}
	// now the constants
	if (params.num_constants) {
		cutting_pct = rand() % params.num_constants;
		for (int i = 0; i < cutting_pct; i++) {
			offspring1.constants[i] = parent1.constants[i];
			offspring2.constants[i] = parent2.constants[i];
		}
		for (int i = cutting_pct; i < params.num_constants; i++) {
			offspring1.constants[i] = parent2.constants[i];
			offspring2.constants[i] = parent1.constants[i];
		}
	}
}
//---------------------------------------------------------------------------
void uniform_crossover(const t_mep_chromosome& parent1, const t_mep_chromosome& parent2,
	const t_mep_parameters& params,
	t_mep_chromosome& offspring1, t_mep_chromosome& offspring2)
{
	for (int i = 0; i < params.code_length; i++)
		if (rand() % 2) {
			offspring1.code[i] = parent1.code[i];
			offspring2.code[i] = parent2.code[i];
		}
		else {
			offspring1.code[i] = parent2.code[i];
			offspring2.code[i] = parent1.code[i];
		}

	// constants
	for (int i = 0; i < params.num_constants; i++)
		if (rand() % 2) {
			offspring1.constants[i] = parent1.constants[i];
			offspring2.constants[i] = parent2.constants[i];
		}
		else {
			offspring1.constants[i] = parent2.constants[i];
			offspring2.constants[i] = parent1.constants[i];
		}
}
//---------------------------------------------------------------------------
int sort_function(const void* a, const void* b)
{// comparator for quick sort
	if (((t_mep_chromosome*)a)->fitness > ((t_mep_chromosome*)b)->fitness)
		return 1;
	else
		if (((t_mep_chromosome*)a)->fitness < ((t_mep_chromosome*)b)->fitness)
			return -1;
		else
			return 0;
}
//---------------------------------------------------------------------------
void print_chromosome(const t_mep_chromosome& a, const t_mep_parameters& params, int num_variables)
{
	printf("The chromosome is:\n");
	printf("//------------------------------------------\n");
	for (int i = 0; i < params.num_constants; i++)
		printf("constants[%d] = %lf\n", i, a.constants[i]);

	for (int i = 0; i < params.code_length; i++) {
		if (a.code[i].op < 0) {
			if (a.code[i].op == SIN_OP)
				printf("%d: sin %d\n", i, a.code[i].addr1);
			else// binary operators
				printf("%d: %c %d %d\n", i, operators_string[abs(a.code[i].op) - 1], a.code[i].addr1, a.code[i].addr2);
		}
		else {
			if (a.code[i].op < num_variables)
				printf("%d: inputs[%d]\n", i, a.code[i].op);
			else
				printf("%d: constants[%d]\n", i, a.code[i].op - num_variables);
		}
	}
	printf("//------------------------------------------\n");
	printf("Best index (output provider) = %d\n", a.best_index);
	printf("Fitness = %lf\n", a.fitness);
}
//---------------------------------------------------------------------------
void print_chromosomeline(const t_mep_chromosome& a, const t_mep_parameters& params, int num_variables, int t)
{
	printf("The chromosome is:\n");
	printf("//------------------------------------------\n");
	for (int i = 0; i < params.num_constants; i++)
		printf("constants[%d] = %lf\n", i, a.constants[i]);

	for (int i = 0; i < (params.code_length - t - 1); i++) {
		if (a.code[i].op < 0) {
			if (a.code[i].op == SIN_OP)
				printf("%d: sin %d\n", i, a.code[i].addr1);
			else// binary operators
				printf("%d: %c %d %d\n", i, operators_string[abs(a.code[i].op) - 1], a.code[i].addr1, a.code[i].addr2);
		}
		else {
			if (a.code[i].op < num_variables)
				printf("%d: inputs[%d]\n", i, a.code[i].op);
			else
				printf("%d: constants[%d]\n", i, a.code[i].op - num_variables);
		}
	}
	printf("//------------------------------------------\n");
	printf("Best index (output provider) = %d\n", a.best_index);
	printf("Fitness = %lf\n", a.fitness);
}
#include <stdlib.h>
#include <time.h>

int roulette_selection(const t_mep_chromosome* population, int pop_size) {
	// 计算总的适应度
	double total_fitness = 0.0;
	for (int i = 0; i < pop_size; i++) {
		total_fitness += population[i].fitness;
	}

	// 生成一个[0, total_fitness)之间的随机数
	double spin = (double)rand() / RAND_MAX * total_fitness;

	// 根据随机数选择个体
	double partial_sum = 0.0;
	for (int i = 0; i < pop_size; i++) {
		partial_sum += population[i].fitness;
		if (spin < partial_sum) {
			return i; // 返回选中的个体的索引
		}
	}

	// 如果由于浮点数精度问题没有返回，返回最后一个个体
	return pop_size - 1;
}

// 注意：在使用轮盘赌选择之前，需要初始化随机数生成器
//---------------------------------------------------------------------------

t_mep_chromosome start_steady_state_mep(t_mep_parameters& params,
	const double** training_data, const double* target, int num_training_data, int num_variables)
{
	// a steady state approach:
	// we work with 1 (one) population
	// newly created individuals will replace the worst existing ones (only if the offspring are better).

	// allocate memory
	t_mep_chromosome* population;
	population = new t_mep_chromosome[params.pop_size];
	for (int i = 0; i < params.pop_size; i++)
		allocate_chromosome(population[i], params);

	t_mep_chromosome offspring1, offspring2;
	allocate_chromosome(offspring1, params);
	allocate_chromosome(offspring2, params);

	double** eval_matrix;
	allocate_partial_expression_values(eval_matrix, num_training_data, params.code_length);

	// initialize
	for (int i = 0; i < params.pop_size; i++) {
		generate_random_chromosome(population[i], params, num_variables);
		if (params.problem_type == PROBLEM_REGRESSION)
			fitness_regression(population[i], params.code_length, num_variables, num_training_data, training_data, target, eval_matrix);
		else// classification problem
			fitness_classification(population[i], params, num_variables, num_training_data, training_data, target, eval_matrix);
	}
	// sort ascendingly by fitness
	qsort((void*)population, params.pop_size, sizeof(population[0]), sort_function);

	printf("generation %d, best fitness = %lf\n", 0, population[0].fitness);

	for (int generation = 1; generation < params.num_generations; generation++) {// for each generation
		for (int k = 0; k < params.pop_size; k += 2) {
			// choose the parents using binary tournament
			int r1 = roulette_selection(population, 2);
			int r2 = roulette_selection(population, 2);
			// crossover
			double p = rand() / double(RAND_MAX);
			if (p < params.crossover_probability)
				one_cut_point_crossover(population[r1], population[r2], params, offspring1, offspring2);
			else {// no crossover so the offspring are a copy of the parents
				copy_individual(offspring1, population[r1], params);
				copy_individual(offspring2, population[r2], params);
			}
			// mutate the result and compute fitness
			mutation(offspring1, params, num_variables);
			if (params.problem_type == PROBLEM_REGRESSION)
				fitness_regression(offspring1, params.code_length, num_variables, num_training_data, training_data, target, eval_matrix);
			else// classification problem
				fitness_classification(offspring1, params, num_variables, num_training_data, training_data, target, eval_matrix);
			// mutate the other offspring and compute fitness
			mutation(offspring2, params, num_variables);
			if (params.problem_type == PROBLEM_REGRESSION)
				fitness_regression(offspring2, params.code_length, num_variables, num_training_data, training_data, target, eval_matrix);
			else // classification problem
				fitness_classification(offspring2, params, num_variables, num_training_data, training_data, target, eval_matrix);

			// replace the worst in the population
			if (offspring1.fitness < population[params.pop_size - 1].fitness) {
				copy_individual(population[params.pop_size - 1], offspring1, params);
				qsort((void*)population, params.pop_size, sizeof(population[0]), sort_function);
			}
			if (offspring2.fitness < population[params.pop_size - 1].fitness) {
				copy_individual(population[params.pop_size - 1], offspring2, params);
				qsort((void*)population, params.pop_size, sizeof(population[0]), sort_function);
			}
		}
		printf("generation %d, best fitness = %lf\n", generation, population[0].fitness);
	}
	// print best chromosome which is always on position 0 because of sorting
	print_chromosome(population[0], params, num_variables);
	t_mep_chromosome best;
	allocate_chromosome(best, params);
	best = population[0];

	delete_chromosome(offspring1);
	delete_chromosome(offspring2);

	for (int i = 0; i < params.pop_size; i++)
		delete_chromosome(population[i]);
	delete[] population;

	delete_partial_expression_values(eval_matrix, params.code_length);
	return best;
}
//--------------------------------------------------------------------
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

	char* buf = new char[10000];
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
	allocate_training_data(data, target, num_data, num_variables);
	for (int i = 0; i < num_data; i++) {
		for (int j = 0; j < num_variables; j++)
			fscanf_s(f, "%lf", &data[i][j]);
		fscanf_s(f, "%lf", &target[i]);
	}
	fclose(f);
	return true;
}
//---------------------------------------------------------------------------


// 主函数

int main(void)
{
	t_mep_parameters params;

	params.pop_size = 80;						    // the number of individuals in population  (must be an even number!)
	params.code_length = 40;
	params.num_generations = 1000;					// the number of generations
	params.mutation_probability = 0.05;              // mutation probability
	params.crossover_probability = 0.6;             // crossover probability

	params.variables_probability = 0.4;
	params.operators_probability = 0.5;
	params.constants_probability = 1 - params.variables_probability - params.operators_probability; // sum of variables_prob + operators_prob + constants_prob MUST BE 1 !

	params.num_constants = 10; // use 5 constants from -1 ... +1 interval
	params.constants_min = -5;
	params.constants_max = 5;

	params.problem_type = PROBLEM_REGRESSION; //0 - regression, 1 - classification; DONT FORGET TO SET IT
	params.classification_threshold = 0; // only for classification problems
	int value = 0;
	int  num_training_data;
	int  num_variables;
	double** training_data, * target;

	if (!read_data("C:/Users/lenovo/Desktop/论文1数据/synthetic_dataset_5.txt", ' ', training_data, target, num_training_data, num_variables)) {

		//if (!read_data("datasets/building1.txt", ' ', &training_data, target, num_training_data, num_variables)) {
		printf("Cannot find file! Please specify the full path!");
		getchar();
		return 1;
	}
	srand(0);// random seed
	clock_t start_time = clock();

	double mid = 0;

	t_mep_chromosome best;
	best.code = NULL;
	best.constants = NULL;
	best.best_index = 0;
	best.fitness = 0;
	best = start_steady_state_mep(params, (const double**)training_data, (const double*)target, num_training_data, num_variables);
	t_mep_chromosome best_chromosome = best;
	double numm = mid / 10;
	printf("%f", numm);

	clock_t end_time = clock();

	double running_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

	printf("Running time = %lf seconds\n", running_time);

	delete_training_data(training_data, target, num_training_data);

	printf("Press enter ...");
	getchar();

	return 0;
}

//--------------------------------------------------------------------
