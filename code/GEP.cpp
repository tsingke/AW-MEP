/**************************************************************************
*  GEP quick-replace for MEP  (single file, ready to compile)
*  编译：cl /O2 gep.cpp  或  g++ -O3 gep.cpp -o gep
**************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <float.h>
#include <algorithm>

#define PROBLEM_REGRESSION 0
#define PROBLEM_BINARY_CLASSIFICATION 1

/*---------- 算子 ----------*/
enum { OP_ADD = 0, OP_SUB, OP_MUL, OP_DIV, OP_SIN, OP_COS, OP_TAN, OP_LOG, OP_EXP, OP_SQRT, OP_POW, OP_MIN, OP_MAX, OP_ABS, OP_FLOOR, OP_CEIL, OP_AND, OP_OR, OP_NUM };
static const char* op_str[] = { "+","-","*","/","sin","cos","tan","log","exp","sqrt","pow","min","max","abs","floor","ceil","&","|" };
static const int   op_arity[] = { 2,2,2,2,1,1,1,1,1,1,2,2,2,1,1,1,2,2 };

/*---------- 参数 ----------*/
struct GepParam {
    int head, tail;              // 染色体 = head + tail
    int gene_cnt;                // 每条染色体含几个基因（多基因 = 多输出/子表达式）
    int code_len;                // = (head+tail)*gene_cnt
    int pop_size, generations;
    double mut_rate, cross_rate;
    int    num_var, num_const;
    double const_min, const_max;
    int    problem_type;         // 0=回归 1=二分类
    double class_th;             // 分类阈值
    /* 概率 */
    double p_op, p_var, p_const; // 初始化用，和为1
};

/*---------- 染色体 ----------*/
struct Chrom {
    int* code;                 // 线性染色体
    double* con;                  // 常数池
    double fitness;
    int    best_gene;            // 最优子表达式序号
};

/*---------- 工具 ----------*/
inline double rand01() { return rand() / (double)RAND_MAX; }
inline int    randInt(int a, int b) { return a + rand() % (b - a + 1); }

/*---------- 分配/释放 ----------*/
void alloc_chrom(Chrom& c, const GepParam& p) {
    c.code = new int[p.code_len];
    c.con = new double[p.num_const];
}
void free_chrom(Chrom& c) {
    delete[] c.code; delete[] c.con;
}

/*---------- 解码 ----------*/
struct Node { int op; double c; int left, right; }; // 简单树节点
Node node_buf[2048]; int node_cnt;

int decode_gene(const int* gene, int h, int t, int num_var, int& pos, int& arity_needed)
/* 把 gene[pos..] 展开成一棵树，返回根节点在 node_buf 的下标 */
{
    int idx = node_cnt++;
    Node& nd = node_buf[idx];
    if (pos >= h) { // tail 区，只能是终端
        if (rand01() < 0.5 && num_var > 0) { nd.op = -1; nd.left = randInt(0, num_var - 1); nd.right = -1; }
        else { nd.op = -2; nd.c = 0;                   nd.right = -1; }
        arity_needed = 0;
        return idx;
    }
    int op = gene[pos++];
    if (op < 0) { // 终端
        if (op < -100) { // 常数占位
            nd.op = -2; nd.c = 0; nd.left = nd.right = -1;
        }
        else { // 变量
            nd.op = -1; nd.left = -op - 1; nd.right = -1;
        }
        arity_needed = 0;
    }
    else { // 函数
        nd.op = op; nd.left = nd.right = -1;
        int need1, need2 = 0;
        nd.left = decode_gene(gene, h, t, num_var, pos, need1);
        if (op_arity[op] == 2) nd.right = decode_gene(gene, h, t, num_var, pos, need2);
        arity_needed = 1 + need1 + need2;
    }
    return idx;
}

/*---------- 树求值 ----------*/
double eval_tree(int root, const double* var, const double* con)
{
    const Node& nd = node_buf[root];
    if (nd.op == -1) return var[nd.left];
    if (nd.op == -2) return nd.c;
    double l = eval_tree(nd.left, var, con);
    double r = nd.right >= 0 ? eval_tree(nd.right, var, con) : 0;
    switch (nd.op) {
    case OP_ADD: return l + r;
    case OP_SUB: return l - r;
    case OP_MUL: return l * r;
    case OP_DIV: return fabs(r) < 1e-14 ? 1e14 : l / r;
    case OP_SIN: return sin(l);
    case OP_COS: return cos(l);
    case OP_TAN: return tan(l);
    case OP_LOG: return l > 0 ? log(l) : -1e14;
    case OP_EXP: return exp(l);
    case OP_SQRT:return l >= 0 ? sqrt(l) : 0;
    case OP_POW: return pow(l, r);
    case OP_MIN: return fmin(l, r);
    case OP_MAX: return fmax(l, r);
    case OP_ABS: return fabs(l);
    case OP_FLOOR:return floor(l);
    case OP_CEIL: return ceil(l);
    case OP_AND: return (int)l & (int)r;
    case OP_OR:  return (int)l | (int)r;
    }
    return 0;
}

/*---------- 线性回归调常数 ----------*/
void refine_constants(Chrom& c, const GepParam& p, const double** x, const double* y, int n)
{
    /* 仅对常数占位符做最小二乘，结构不变 */
    static double A[10000][32], b[10000];
    int m = std::min(n, 10000), cols = 0;
    for (int i = 0; i < p.num_const; i++) cols++; // 每个真实常数
    if (cols == 0) return;
    for (int k = 0; k < m; k++) {
        double tmp_var[64];
        for (int v = 0; v < p.num_var; v++) tmp_var[v] = x[k][v];
        node_cnt = 0; int pos = 0, dummy = 0;
        decode_gene(c.code, p.head, p.tail, p.num_var, pos, dummy);
        for (int g = 0; g < p.gene_cnt; g++) {
            int root = decode_gene(c.code + g * (p.head + p.tail), p.head, p.tail, p.num_var, pos, dummy);
            A[k][g] = eval_tree(root, tmp_var, c.con);
        }
        b[k] = y[k];
    }
    /* 正规方程 */
    static double AtA[32][32], Atb[32];
    memset(AtA, 0, sizeof(AtA)); memset(Atb, 0, sizeof(Atb));
    for (int i = 0; i < cols; i++)
        for (int j = 0; j < cols; j++)
            for (int k = 0; k < m; k++) AtA[i][j] += A[k][i] * A[k][j];
    for (int i = 0; i < cols; i++)
        for (int k = 0; k < m; k++) Atb[i] += A[k][i] * b[k];
    /* 高斯消元 */
    for (int i = 0; i < cols; i++) {
        for (int j = i + 1; j < cols; j++) {
            double rate = AtA[j][i] / AtA[i][i];
            for (int k = i; k < cols; k++) AtA[j][k] -= rate * AtA[i][k];
            Atb[j] -= rate * Atb[i];
        }
    }
    for (int i = cols - 1; i >= 0; i--) {
        for (int j = i + 1; j < cols; j++) Atb[i] -= AtA[i][j] * c.con[j];
        c.con[i] = Atb[i] / AtA[i][i];
    }
}

/*---------- 适应度 ----------*/
void fitness(Chrom& c, const GepParam& p, const double** x, const double* y, int n)
{
    double best_err = 1e308; int best_g = 0;
    for (int g = 0; g < p.gene_cnt; g++) {
        double sse = 0;
        for (int k = 0; k < n; k++) {
            double tmp_var[64];
            for (int v = 0; v < p.num_var; v++) tmp_var[v] = x[k][v];
            node_cnt = 0; int pos = 0, dummy = 0;
            int root = decode_gene(c.code + g * (p.head + p.tail), p.head, p.tail, p.num_var, pos, dummy);
            double o = eval_tree(root, tmp_var, c.con);
            double d = o - y[k];
            sse += d * d;
        }
        double err = sse / n;
        if (err < best_err) { best_err = err; best_g = g; }
    }
    c.fitness = best_err;
    c.best_gene = best_g;
}

/*---------- 初始化 ----------*/
void random_chrom(Chrom& c, const GepParam& p)
{
    for (int i = 0; i < p.num_const; i++) c.con[i] = p.const_min + rand01() * (p.const_max - p.const_min);
    for (int g = 0; g < p.gene_cnt; g++) {
        int* gene = c.code + g * (p.head + p.tail);
        /* head */
        for (int i = 0; i < p.head; i++) {
            double r = rand01();
            if (r < p.p_op)      gene[i] = randInt(0, OP_NUM - 1);
            else if (r < p.p_op + p.p_var) gene[i] = -randInt(1, p.num_var) - 1000;
            else              gene[i] = -(p.num_var + randInt(0, p.num_const - 1) + 1) - 2000;
        }
        /* tail */
        for (int i = p.head; i < p.head + p.tail; i++) {
            if (rand01() < 0.5 && p.num_var > 0) gene[i] = -randInt(1, p.num_var) - 1000;
            else                            gene[i] = -(p.num_var + randInt(0, p.num_const - 1) + 1) - 2000;
        }
    }
}

/*---------- 变异 ----------*/
void mutate(Chrom& c, const GepParam& p)
{
    for (int g = 0; g < p.gene_cnt; g++) {
        int* gene = c.code + g * (p.head + p.tail);
        /* head */
        for (int i = 0; i < p.head; i++) if (rand01() < p.mut_rate) {
            double r = rand01();
            if (r < p.p_op)      gene[i] = randInt(0, OP_NUM - 1);
            else if (r < p.p_op + p.p_var) gene[i] = -randInt(1, p.num_var) - 1000;
            else              gene[i] = -(p.num_var + randInt(0, p.num_const - 1) + 1) - 2000;
        }
        /* tail */
        for (int i = p.head; i < p.head + p.tail; i++) if (rand01() < p.mut_rate) {
            if (rand01() < 0.5 && p.num_var > 0) gene[i] = -randInt(1, p.num_var) - 1000;
            else                            gene[i] = -(p.num_var + randInt(0, p.num_const - 1) + 1) - 2000;
        }
    }
    /* 常数 */
    for (int i = 0; i < p.num_const; i++) if (rand01() < p.mut_rate)
        c.con[i] = p.const_min + rand01() * (p.const_max - p.const_min);
}

/*---------- 交叉 ----------*/
void crossover(const Chrom& p1, const Chrom& p2, const GepParam& p, Chrom& o1, Chrom& o2)
{
    /* 单点交叉，按基因切 */
    int cut = randInt(0, p.gene_cnt);
    int gs = p.head + p.tail;
    for (int g = 0; g < cut; g++) {
        memcpy(o1.code + g * gs, p1.code + g * gs, gs * sizeof(int));
        memcpy(o2.code + g * gs, p2.code + g * gs, gs * sizeof(int));
    }
    for (int g = cut; g < p.gene_cnt; g++) {
        memcpy(o1.code + g * gs, p2.code + g * gs, gs * sizeof(int));
        memcpy(o2.code + g * gs, p1.code + g * gs, gs * sizeof(int));
    }
    /* 常数 */
    int cc = randInt(0, p.num_const);
    memcpy(o1.con, p1.con, cc * sizeof(double));
    memcpy(o2.con, p2.con, cc * sizeof(double));
    memcpy(o1.con + cc, p2.con + cc, (p.num_const - cc) * sizeof(double));
    memcpy(o2.con + cc, p1.con + cc, (p.num_const - cc) * sizeof(double));
}

/*---------- 选择 ----------*/
int tournament(const Chrom* pop, int n, int k = 3)
{
    int best = randInt(0, n - 1);
    for (int i = 1; i < k; i++) {
        int idx = randInt(0, n - 1);
        if (pop[idx].fitness < pop[best].fitness) best = idx;
    }
    return best;
}
/*---------- 辅助：染色体拷贝 ----------*/
inline void copy_chrom(Chrom& dst, const Chrom& src, const GepParam& p)
{
    memcpy(dst.code, src.code, p.code_len * sizeof(int));
    memcpy(dst.con, src.con, p.num_const * sizeof(double));
    dst.fitness = src.fitness;
    dst.best_gene = src.best_gene;
}
/*---------- 主循环 ----------*/
Chrom evolve(const GepParam& p, const double** x, const double* y, int n)
{
    Chrom* pop = new Chrom[p.pop_size];
    for (int i = 0; i < p.pop_size; i++) alloc_chrom(pop[i], p);
    Chrom o1, o2; alloc_chrom(o1, p); alloc_chrom(o2, p);

    /* 初始种群 */
    for (int i = 0; i < p.pop_size; i++) {
        random_chrom(pop[i], p);
        refine_constants(pop[i], p, x, y, n);
        fitness(pop[i], p, x, y, n);
    }
    std::sort(pop, pop + p.pop_size, [](const Chrom& a, const Chrom& b) {return a.fitness < b.fitness; });
    printf("gen 0 best=%.6f\n", pop[0].fitness);

    /* 演化 */
    for (int g = 1; g <= p.generations; g++) {
        for (int k = 0; k < p.pop_size; k += 2) {
            int i1 = tournament(pop, p.pop_size);
            int i2 = tournament(pop, p.pop_size);
            copy_chrom(o1, pop[i1], p); copy_chrom(o2, pop[i2], p);
            if (rand01() < p.cross_rate) crossover(pop[i1], pop[i2], p, o1, o2);
            mutate(o1, p); mutate(o2, p);
            refine_constants(o1, p, x, y, n); refine_constants(o2, p, x, y, n);
            fitness(o1, p, x, y, n); fitness(o2, p, x, y, n);
            /* 替代最差的 */
            if (o1.fitness < pop[p.pop_size - 1].fitness) {
                copy_chrom(pop[p.pop_size - 1], o1, p);
                std::sort(pop, pop + p.pop_size, [](const Chrom& a, const Chrom& b) {return a.fitness < b.fitness; });
            }
            if (o2.fitness < pop[p.pop_size - 1].fitness) {
                copy_chrom(pop[p.pop_size - 1], o2, p);
                std::sort(pop, pop + p.pop_size, [](const Chrom& a, const Chrom& b) {return a.fitness < b.fitness; });
            }
        }
        printf("gen %d best=%.6f\n", g, pop[0].fitness);
    }

    Chrom best; alloc_chrom(best, p); copy_chrom(best, pop[0], p);
    for (int i = 0; i < p.pop_size; i++) free_chrom(pop[i]);
    free_chrom(o1); free_chrom(o2);
    delete[] pop;
    return best;
}

/*---------- 文件读取 ----------*/
bool read_data(const char* file, double**& x, double*& y, int& n, int& var)
{
    FILE* f = nullptr;
    errno_t err = fopen_s(&f, file, "r");
    if (err != 0) return false;
    n = 0; var = 0;
    double buf[256];
    while (fscanf_s(f, "%lf", &buf[var]) == 1) {
        var++;
        char c = fgetc(f);
        if (c == '\n' || c == EOF) { if (n == 0) var--; n++; break; }
        ungetc(c, f);
    }
    var++;
    fclose(f);
    x = new double* [n]; for (int i = 0; i < n; i++) x[i] = new double[var - 1];
    y = new double[n];
    err = fopen_s(&f, file, "r");
    if (err != 0) return false;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < var - 1; j++) fscanf_s(f, "%lf", &x[i][j]);
        fscanf_s(f, "%lf", &y[i]);
    }
    fclose(f);
    var--;
    return true;
}

/*---------- 主函数 ----------*/
int main()
{
    srand((unsigned)time(NULL));
    GepParam p;
    p.head = 5;
    p.gene_cnt = 3;
    p.tail = p.head * (2 - 1) + 1;
    p.code_len = (p.head + p.tail) * p.gene_cnt;
    p.pop_size = 80;
    p.generations =1000;
    p.mut_rate = 0.05;
    p.cross_rate = 0.6;
    p.num_const = 10;
    p.const_min = -10;
    p.const_max = 10;
    p.p_op = 0.7; p.p_var = 0.1; p.p_const = 0.2;
    p.problem_type = PROBLEM_REGRESSION;

    double** x; double* y; int n, var;
    if (!read_data("C:/Users/lenovo/Desktop/论文1数据/synthetic_dataset_5.txt", x, y, n, var)) { puts("file error"); return 0; }
    p.num_var = var;

    Chrom best = evolve(p, const_cast<const double**>(x), y, n);
    printf("\nBest train RMSE=%.6f  gene=%d\n", sqrt(best.fitness), best.best_gene);
    /* 打印表达式略，可自己展开 best.code */

    for (int i = 0; i < n; i++) delete[] x[i];
    delete[] x; delete[] y;
    return 0;
}
