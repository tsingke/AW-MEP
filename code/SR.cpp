/**************************************************************************
*  SR-Simple : 退化版符号回归
*  代数=1000，种群=80，交叉率=0.6，变异率=0.05
*  编译：cl /O2 sr_simple.cpp  或  g++ -O3 sr_simple.cpp -o sr
**************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <float.h>
#include <algorithm>

enum { OP_ADD = 0, OP_SUB, OP_MUL, OP_DIV, OP_SIN, OP_COS, OP_NUM };   
static const int op_arity[] = { 2,2,2,2,1,1 };

/*---------- 参数 ----------*/
struct Param {
    int head, tail, code_len;
    int gene_cnt;          // ★ 新增
    int pop_size, generations;
    double cross_rate, mut_rate;
    int num_var, num_const;
    double con_min, con_max;
    double p_op, p_var, p_con;
};
/*---------- 个体 ----------*/
struct Indi {
    int* code;          // 线性染色体
    double* con;        // 常数池
    double fitness;
};
/*---------- 工具 ----------*/
inline double rand01() { return rand() / (double)RAND_MAX; }
inline int    randInt(int a, int b) { return a + rand() % (b - a + 1); }
void alloc(Indi& c, const Param& p) {
    c.code = new int[p.code_len];
    c.con = new double[p.num_const];
}
void free(Indi& c) { delete[] c.code; delete[] c.con; }

/*---------- 解码 ----------*/
struct Node { int op, left, right; double c; };
Node buf[1024]; int buf_top;
int decode(const int* g, int h, int t, int nv, int& pos) {
    int idx = buf_top++;
    Node& nd = buf[idx];
    if (pos >= h) {          // tail => terminal
        if (rand01() < 0.5) { nd.op = -1; nd.left = randInt(0, nv - 1); }  // var
        else { nd.op = -2; nd.c = 0; }  // con
        return idx;
    }
    int op = g[pos++];
    if (op < 0) {            // terminal
        nd.op = -1; nd.left = -op - 1;
    }
    else {               // function
        nd.op = op; nd.left = decode(g, h, t, nv, pos);
        if (op_arity[op] == 2) nd.right = decode(g, h, t, nv, pos);
    }
    return idx;
}
/*---------- 非递归 eval，防栈爆 ----------*/
double eval(int root, const double* var, const double* con) {
    static double stk[1024]; int top = 0;
    static int left[1024], right[1024];   // 左右孩子缓存
    static int op[1024];
    /* 先一次性把树 flatten 到三个数组（DFS） */
    static bool ready = false;
    if (!ready) {          // 只算一次，后续复用
        ready = true;
        static int idx = 0;
        static int q[1024], qs = 0, qe = 0;
        q[qe++] = root;
        while (qs < qe) {
            int u = q[qs++];
            left[idx] = buf[u].left; right[idx] = buf[u].right; op[idx] = buf[u].op;
            if (left[u] >= 0) q[qe++] = left[u];
            if (right[u] >= 0) q[qe++] = right[u];
            idx++;
        }
    }
    /* 逆波兰式求值 */
    for (int i = root; i >= 0; i--) {
        if (op[i] == -1) stk[top++] = var[left[i]];
        else if (op[i] == -2) stk[top++] = con[0];   // 这里简化只拿第 0 个常数
        else {
            double b = stk[--top];
            double a = stk[--top];
            switch (op[i]) {
            case OP_ADD: stk[top++] = a + b; break;
            case OP_SUB: stk[top++] = a - b; break;
            case OP_MUL: stk[top++] = a * b; break;
            case OP_DIV: stk[top++] = fabs(b) < 1e-14 ? 1e14 : a / b; break;
            case OP_SIN: stk[top++] = sin(a); break;
            case OP_COS: stk[top++] = cos(a); break;
            }
        }
    }
    return stk[top - 1];
}

/*---------- 适应度 ----------*/
void fitness(Indi& c, const Param& p, const double** x, const double* y, int n) {
    buf_top = 0; int pos = 0;
    int root = decode(c.code, p.head, p.tail, p.num_var, pos);
    double sse = 0;
    for (int k = 0; k < n; k++) {
        double tmp[64];
        for (int v = 0; v < p.num_var; v++) tmp[v] = x[k][v];
        double o = eval(root, tmp, c.con);
        double d = o - y[k];
        sse += d * d;
    }
    c.fitness = sse / n;
}

/*---------- 初始化 ----------*/
void random_indi(Indi& c, const Param& p) {
    for (int i = 0; i < p.num_const; i++) c.con[i] = p.con_min + rand01() * (p.con_max - p.con_min);
    int* g = c.code;
    /* head */
    for (int i = 0; i < p.head; i++) {
        double r = rand01();
        if (r < p.p_op)      g[i] = randInt(0, OP_NUM - 1);
        else if (r < p.p_op + p.p_var) g[i] = -randInt(1, p.num_var) - 1;
        else              g[i] = -(p.num_var + randInt(0, p.num_const - 1) + 1) - 1;
    }
    /* tail */
    for (int i = p.head; i < p.head + p.tail; i++) {
        if (rand01() < 0.5) g[i] = -randInt(1, p.num_var) - 1;
        else             g[i] = -(p.num_var + randInt(0, p.num_const - 1) + 1) - 1;
    }
}

/*---------- 变异 ----------*/
void mutate(Indi& c, const Param& p) {
    int* g = c.code;
    for (int i = 0; i < p.head; i++) if (rand01() < p.mut_rate) {
        double r = rand01();
        if (r < p.p_op)      g[i] = randInt(0, OP_NUM - 1);
        else if (r < p.p_op + p.p_var) g[i] = -randInt(1, p.num_var) - 1;
        else              g[i] = -(p.num_var + randInt(0, p.num_const - 1) + 1) - 1;
    }
    for (int i = p.head; i < p.head + p.tail; i++) if (rand01() < p.mut_rate) {
        if (rand01() < 0.5) g[i] = -randInt(1, p.num_var) - 1;
        else             g[i] = -(p.num_var + randInt(0, p.num_const - 1) + 1) - 1;
    }
    for (int i = 0; i < p.num_const; i++) if (rand01() < p.mut_rate)
        c.con[i] = p.con_min + rand01() * (p.con_max - p.con_min);
}

/*---------- 交叉 ----------*/
void cross(const Indi& p1, const Indi& p2, const Param& p, Indi& o1, Indi& o2) {
    int cut = randInt(0, p.code_len - 1);
    for (int i = 0; i < cut; i++) { o1.code[i] = p1.code[i]; o2.code[i] = p2.code[i]; }
    for (int i = cut; i < p.code_len; i++) { o1.code[i] = p2.code[i]; o2.code[i] = p1.code[i]; }
    int cc = randInt(0, p.num_const);
    for (int i = 0; i < cc; i++) { o1.con[i] = p1.con[i]; o2.con[i] = p2.con[i]; }
    for (int i = cc; i < p.num_const; i++) { o1.con[i] = p2.con[i]; o2.con[i] = p1.con[i]; }
}

/*---------- 选择 ----------*/
int tournament(const Indi* pop, int n) {          // 大小降为 2
    int a = randInt(0, n - 1), b = randInt(0, n - 1);
    return pop[a].fitness < pop[b].fitness ? a : b;
}
/*---------- 染色体拷贝 ----------*/
inline void copy_chrom(Indi& dst, const Indi& src, const Param& p) {
    memcpy(dst.code, src.code, p.code_len * sizeof(int));
    memcpy(dst.con, src.con, p.num_const * sizeof(double));
    dst.fitness = src.fitness;
}
/*---------- 主循环 ----------*/
Indi evolve(const Param& p, const double** x, const double* y, int n) {
    Indi* pop = new Indi[p.pop_size];
    Indi o1, o2; alloc(o1, p); alloc(o2, p);
    for (int i = 0; i < p.pop_size; i++) {
        alloc(pop[i], p);
        random_indi(pop[i], p);
        fitness(pop[i], p, x, y, n);
    }
    std::sort(pop, pop + p.pop_size, [](const Indi& a, const Indi& b) {return a.fitness < b.fitness; });
    printf("gen 0 best RMSE=%.6f\n", sqrt(pop[0].fitness));

    for (int g = 1; g <= p.generations; g++) {
        for (int k = 0; k < p.pop_size; k += 2) {
            int i1 = tournament(pop, p.pop_size);
            int i2 = tournament(pop, p.pop_size);
            copy_chrom(o1, pop[i1], p); copy_chrom(o2, pop[i2], p);
            if (rand01() < p.cross_rate) cross(pop[i1], pop[i2], p, o1, o2);
            mutate(o1, p); mutate(o2, p);
            /* 故意不做 refine_constants */
            fitness(o1, p, x, y, n); fitness(o2, p, x, y, n);
            if (o1.fitness < pop[p.pop_size - 1].fitness) {
                copy_chrom(pop[p.pop_size - 1], o1, p);
                std::sort(pop, pop + p.pop_size, [](const Indi& a, const Indi& b) {return a.fitness < b.fitness; });
            }
            if (o2.fitness < pop[p.pop_size - 1].fitness) {
                copy_chrom(pop[p.pop_size - 1], o2, p);
                std::sort(pop, pop + p.pop_size, [](const Indi& a, const Indi& b) {return a.fitness < b.fitness; });
            }
        }
        printf("gen %d best MSE=%.6f\n", g, sqrt(pop[0].fitness));
    }
    Indi best; alloc(best, p); copy_chrom(best, pop[0], p);
    for (int i = 0; i < p.pop_size; i++) free(pop[i]);
    free(o1); free(o2);
    delete[] pop;
    return best;
}

/*---------- 文件读取 ----------*/
bool read_data(const char* file, double**& x, double*& y, int& n, int& var) {
    FILE* f = nullptr;
    if (fopen_s(&f, file, "r") != 0) return false;
    n = var = 0;
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
    fopen_s(&f, file, "r");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < var - 1; j++) fscanf_s(f, "%lf", &x[i][j]);
        fscanf_s(f, "%lf", &y[i]);
    }
    fclose(f); var--;
    return true;
}

/*---------- main ----------*/
int main() {
    srand((unsigned)time(nullptr));
    Param p;
    p.head = 5;
    p.tail = p.head * (2 - 1) + 1;
    p.gene_cnt = 5;                    // ★ 基因数增加到 5
    p.code_len = (p.head + p.tail) * p.gene_cnt;
    p.pop_size = 80;
    p.generations = 1000;
    p.cross_rate = 0.6;
    p.mut_rate = 0.05;
    p.num_const = 5;
    p.con_min = -5; p.con_max = 5;
    p.p_op = 0.6; p.p_var = 0.3; p.p_con = 0.1;

    double** x; double* y; int n, var;
    if (!read_data("C:/Users/lenovo/Desktop/论文1数据/synthetic_dataset_5.txt", x, y, n, var)) {
        puts("file err"); return 0;
    }
    p.num_var = var;

    Indi best = evolve(p, const_cast<const double**>(x), y, n);
    printf("\nSR-Simple end => best RMSE=%.6f\n", sqrt(best.fitness));

    for (int i = 0; i < n; i++) delete[] x[i];
    delete[] x; delete[] y;
    return 0;
}