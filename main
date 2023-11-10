#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <random>
#include <omp.h>
#include<gl/glut.h>//加入灵魂

const double PI = 3.14159265;
std::random_device rd; // 随机设备用于获取种子
std::mt19937 gen(rd()); // 梅森旋转算法引擎
std::uniform_real_distribution<double> dis(0.0, 1.0); // 在0到1之间均匀分布double

struct Vec {
    double x, y, z;
    Vec(double x_ = 0, double y_ = 0, double z_ = 0) { x = x_; y = y_; z = z_; }
    Vec operator+(const Vec& b) const { return Vec(x + b.x, y + b.y, z + b.z); }
    Vec operator-(const Vec& b) const { return Vec(x - b.x, y - b.y, z - b.z); }
    Vec operator/(const Vec& b) const { return Vec(x / b.x, y / b.y, z / b.z); }
    Vec operator*(double b) const { return Vec(x * b, y * b, z * b); }
    bool operator==(const Vec& b) const { return x == b.x and y == b.y and z == b.z; }
    Vec mult(const Vec& b) const { return Vec(x * b.x, y * b.y, z * b.z); }
    Vec& norm() { return *this = *this * (1 / sqrt(x * x + y * y + z * z)); }
    double dot(const Vec& b) const { return x * b.x + y * b.y + z * b.z; }
    Vec operator%(Vec& b) { return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
};

struct Matrix {
    Vec v1, v2, v3;//列向量
    Matrix(Vec v1_ = Vec(), Vec v2_ = Vec(), Vec v3_ = Vec()) { v1 = v1_, v2 = v2_, v3 = v3_; }
    Vec Matrix_dot(const Vec& v) const { return Vec(Vec(v1.x, v2.x, v3.x).dot(v), Vec(v1.y, v2.y, v3.y).dot(v), Vec(v1.z, v2.z, v3.z).dot(v)); }
    Matrix Matrix_Multiplication(const Matrix& M) const { return Matrix(Matrix(v1, v2, v3).Matrix_dot(M.v1), Matrix(v1, v2, v3).Matrix_dot(M.v2), Matrix(v1, v2, v3).Matrix_dot(M.v3)); }
};

inline void print(Matrix M) {
    std::cout << "| " << M.v1.x << " " << M.v2.x << " " << M.v3.x << " |\n|" << M.v1.y << " " << M.v2.y << " " << M.v3.y << " |\n|" << M.v1.z << " " << M.v2.z << " " << M.v3.z << " |\n";
}

inline Vec abs(Vec v) {
    return Vec(abs(v.x), abs(v.y), abs(v.z));
}

inline Matrix Euler_Rotation_matrix(double anglex, double angley, double anglez) {
    double angle;
    double sin_zetta;
    double cos_zetta;

    angle = anglex;
    sin_zetta = sin(angle);
    cos_zetta = cos(angle);
    Vec v1(1, 0, 0), v2(0, cos_zetta, sin_zetta), v3(0, -sin_zetta, cos_zetta);
    Matrix x_rotate_matrix(v1, v2, v3);

    angle = angley;
    sin_zetta = sin(angle);
    cos_zetta = cos(angle);
    v1 = Vec(cos_zetta, 0, -sin_zetta), v2 = Vec(0, 1, 0), v3 = Vec(sin_zetta, 0, cos_zetta);
    Matrix y_rotate_matrix(v1, v2, v3);

    angle = anglez;
    sin_zetta = sin(angle);
    cos_zetta = cos(angle);
    v1 = Vec(cos_zetta, sin_zetta, 0), v2 = Vec(-sin_zetta, cos_zetta, 0), v3 = Vec(0, 0, 1);
    Matrix z_rotate_matrix(v1, v2, v3);
    return z_rotate_matrix.Matrix_Multiplication((y_rotate_matrix.Matrix_Multiplication(x_rotate_matrix)));
}

inline Matrix Rotation_matrix(Vec v, double angle) {
    double x = v.x;
    double y = v.y;
    double z = v.z;

    double sin_zetta = sin(angle);
    double cos_zetta = cos(angle);
    double one_sub_cos_zetta = 1 - cos_zetta;
    return Matrix(Vec(cos_zetta + one_sub_cos_zetta * x * x, one_sub_cos_zetta * x * y + sin_zetta * z, one_sub_cos_zetta * x * z - sin_zetta * y),
        Vec(one_sub_cos_zetta * x * y - sin_zetta * z, cos_zetta + one_sub_cos_zetta * y * y, one_sub_cos_zetta * y * z + sin_zetta * x),
        Vec(one_sub_cos_zetta * x * z + sin_zetta * y, one_sub_cos_zetta * y * z - sin_zetta * x, cos_zetta + one_sub_cos_zetta * z * z));
}

struct Ray {
    Vec o, d; Ray(Vec o_, Vec d_) : o(o_), d(d_) {};
};
struct Sphere {
    Vec p, c, e;//位置，反照系数，自发光
    double rad, roughness, transmission_probability, refractivity;//半径，粗糙度,透射概率,折射率
    Sphere(double rad_, Vec p_, Vec e_, Vec c_, double roughness_,double transmission_probability_,double refractivity_) :
        rad(rad_), p(p_), e(e_), c(c_), roughness(roughness_), transmission_probability(transmission_probability_), refractivity(refractivity_){}
    double intersect(const Ray& r) const {
        Vec op = p - r.o;
        double t, eps = 1e-3, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
        if (det < 0) return 0; else det = sqrt(det);
        return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
    }
};
struct Point_Light {//点光
    Vec p, e;
    double rad;
    Point_Light(double rad_, Vec p_, Vec e_) :
        rad(rad_), p(p_), e(e_) {};
};

struct Sun_Light {//方向光
    Vec d, e;//发射方向，强度
    double angle;//弧度
    Sun_Light(double angle_, Vec d_, Vec e_) :
        angle(angle_), d(d_), e(e_) {};
};

Sphere spheres[] = {
      Sphere(1e10, Vec(0,-1e10,0),Vec(),  Vec(1,1,1),.06,0,1.45),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1),
      Sphere(0, Vec(),Vec(),  Vec(),0,0,1)
    
};

Point_Light point_Lights[] = {
    Point_Light(1, Vec(0,30,0),Vec(1,1,1) * 0),
    Point_Light(0, Vec(),Vec() * 0),
    Point_Light(0, Vec(),Vec() * 0),
    Point_Light(0, Vec(),Vec() * 0),
    Point_Light(0, Vec(),Vec() * 0),
    Point_Light(0, Vec(),Vec() * 0),
    Point_Light(0, Vec(),Vec() * 0),
    Point_Light(0, Vec(),Vec() * 0),
    Point_Light(0, Vec(),Vec() * 0),
    //Point_Light(1.5, Vec(20,20,-5),Vec(.9,.6,.65) * 3300),
    //Point_Light(1.5, Vec(0,20,15),Vec(.65,.6,.9) * 3300)
};

Sun_Light sun_Lights[] = {
      Sun_Light(1, Vec(0,-1,0),Vec(.8,.8,.8) * 10)
};

inline bool intersect(const Ray& r, double& t, int& id) {
    double n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
    for (int i = int(n); i--;) {
        if ((d = spheres[i].intersect(r)) && d < t) {
            t = d; id = i;
        }
    }
    return t < inf;
}


inline Vec Vec_pow(const Vec& f, double strength) {
    return Vec(pow(f.x, strength), pow(f.y, strength), pow(f.z, strength));
}



//重要性采样
inline Vec light_MIS(const Vec& x, const Vec& normal, const double& roughness) {
    double n = static_cast<double>(sizeof(point_Lights)) / sizeof(Point_Light);
    Vec sum_res;
    //投射阴影射线
    for (int i = int(n); i--;) {
        double t = 0, inf = 1e20;
        int id = 0;
        double f=1;//透射系数
        double str = 1;//透射强度
        Vec c(0,0,0);
        Vec x2light = point_Lights[i].p - x;
        Vec x2light_norm = (point_Lights[i].p - x).norm();
        Vec Tc(1, 1, 1);//透射近似
        intersect(Ray(x, x2light_norm), t, id);
        double x2light_pow2 = x2light.dot(x2light);
        if (t < sqrt(x2light_pow2) and t > 1e-5) {
            f = spheres[id].transmission_probability;
            c = spheres[id].c;
            str = (spheres[id].p-x).norm().dot(x2light.norm());//模拟焦散
            str = (str < 0) ? 0 : str;
            Tc = Vec_pow(c, pow(1 - f, 2))* str;//透射颜色
            Tc = Vec(1, 1, 1) * (1 - f) + Tc * f;
            //continue;
        }
        double S = pow(point_Lights[i].rad, 2) * PI;
        double oneover_rad = 1.0 / point_Lights[i].rad;
        double shadowing = x2light_norm.dot(normal) * pow(roughness, 1.05);//阴影遮蔽项
        shadowing = (shadowing < 0) ? 0 : shadowing;
        sum_res = sum_res + Tc.mult( point_Lights[i].e * shadowing * (1.0 / (x2light_pow2<1?1: x2light_pow2)) * oneover_rad);////立体角积分
        //Vec_pow(c, pow(1 - f, stranth))项为透射近似
    }
    return sum_res;
}

inline double Vec_max(const Vec& v) {
    return (v.x > v.y) ? ((v.x > v.z) ? v.x : v.z) : ((v.y > v.z) ? v.y : v.z);
}

inline Vec initi_Micro_surface(const Vec& n, const Vec& b_n, const double& strength) {//初始化微表面
    double random_nx = dis(gen) * (PI * 0.5) * strength;
    double random_ny = dis(gen) * (PI * 2);
    Vec n_Micro_surface = Rotation_matrix(n, random_ny).Matrix_dot(Rotation_matrix(b_n, random_nx).Matrix_dot(n));
    n_Micro_surface = n_Micro_surface.norm();
    return n_Micro_surface;
}


inline Vec Reflections(Vec& ref, Vec& micro_normal) {//微表面反射
    return ref - micro_normal * 2 * micro_normal.dot(ref);
}
inline Vec Transmission(Vec& tra, Vec& micro_normal,double Refractive_ratio_inv) {//微表面折射
    return (tra - micro_normal * micro_normal.dot(tra)) * Refractive_ratio_inv - micro_normal * pow(1 - pow(Refractive_ratio_inv, 2) * (1 - pow(tra.dot(micro_normal), 2)), 0.5);//初始化折射方向
}

inline Vec Multiple_reflections(Vec& ref, Vec& micro_normal, const Vec& normal, const Vec& b_normal, const double& roughness, int& reflect_deep) {//微表面多重反射(反射方向，微表面法线，法线)
    ref = Reflections(ref, micro_normal);
    reflect_deep++;
    if (normal.dot(ref) > 0) {
        return ref;
    }
    if (reflect_deep > 6) {
        return Vec();//返回零矢量
    }
    micro_normal = initi_Micro_surface(normal, b_normal, roughness);//重新初始化微表面
    return Multiple_reflections(ref, micro_normal, normal, b_normal, roughness, reflect_deep);
}
inline Vec Multiple_transmission(Vec& tra, Vec& micro_normal, const Vec& normal, const Vec& b_normal, const double& roughness, int& transmission_deep,double& Refractive_ratio_inv) {//微表面多重折射(折射方向，微表面法线，法线)
    tra = Transmission(tra, micro_normal, Refractive_ratio_inv);
    transmission_deep++;
    if (normal.dot(tra) < 0) {
        return tra;
    }
    if (transmission_deep > 6) {
        return Vec();//返回零矢量
    }
    micro_normal = initi_Micro_surface(normal, b_normal, roughness);//重新初始化微表面
    return Multiple_transmission(tra, micro_normal, normal, b_normal, roughness, transmission_deep, Refractive_ratio_inv);
}

inline double color_light(Vec color) {
    return color.x * .27 + color.y * .67 + color.z * .06;
}


inline void hsv_to_rgb(int h, int s, int v, double* R, double* G, double* B)//HSV2RGB
{
    double C = 0, X = 0, Y = 0, Z = 0;
    int i = 0;
    double H = (double)(h);
    double S = (double)(s) / 100.0;
    double V = (double)(v) / 100.0;
    if (S == 0)
        *R = *G = *B = V;
    else
    {
        H = H / 60;
        i = (int)H;
        C = H - i;

        X = V * (1 - S);
        Y = V * (1 - S * C);
        Z = V * (1 - S * (1 - C));
        switch (i) {
        case 0: *R = V; *G = Z; *B = X; break;
        case 1: *R = Y; *G = V; *B = X; break;
        case 2: *R = X; *G = V; *B = Z; break;
        case 3: *R = X; *G = Y; *B = V; break;
        case 4: *R = Z; *G = X; *B = V; break;
        case 5: *R = V; *G = X; *B = Y; break;
        }
    }
}

int min_deepth = 1;
inline Vec radiance(const Ray& r, int depth) {//辐射
    double t;
    int id = 0;
    Vec c1 = Vec(.7, .7, .8);
    Vec c2 = Vec(.9, .8, .9);
    if (!intersect(r, t, id)) {
        double shader_sky = r.d.y*.2;
        return (c1 *(1- shader_sky) + c2 * shader_sky)*.25;//背景
    }
    const Sphere& obj = spheres[id];//obj
    Vec x = r.o + r.d * t;
    Vec n = (x - obj.p).norm();

    Vec projection = n.mult(Vec(1, 0, 1));// 降y维

    Vec tangent = (n % projection).norm(); //切线
    Vec bitangent = (n % tangent).norm();//副切线

    std::random_device rd; // 随机设备用于获取种子
    std::mt19937 gen(rd()); // 梅森旋转算法引擎
    std::uniform_real_distribution<double> dis(0, 1);
    double random_nx = dis(gen) * (PI * 0.5) * obj.roughness;
    double random_ny = dis(gen) * (PI * 2);
    Vec f_ref = obj.c;//反照系数
    Vec f_tra = obj.c;//折射系数

    //shader:
    if (id ==0) {
        double shader_x= fmod(int(x.x+1e5) + int(x.z+1e5), 2);
        f_ref = Vec(shader_x, shader_x, shader_x);
        f_tra = f_ref;
    }



    double p = Vec_max(f_ref);//初始化概率
    ++depth;
    Vec nl = (n.dot(r.d) > 0) ? n * -1 : n;
    double roughness = obj.roughness;
    Vec n_Micro_surface = initi_Micro_surface(nl, bitangent, roughness);//初始化微法线
    //n_Micro_surface = (n.dot(r.d) > 0) ? n_Micro_surface * -1 : n_Micro_surface;
    //double lanbort = nl.dot(r.d) * -1;

    double Environment_refractivity = 1.0005;//环境折射率
    double refractivity = obj.refractivity;
    double transmission_probability = obj.transmission_probability;//光线透过概率
    double Refractive_ratio = refractivity / Environment_refractivity;
    double Refractive_ratio_inv = Environment_refractivity/ refractivity;
    double coszetta = n_Micro_surface.dot(r.d)*-1;
    double Reflectance_Scale = pow((Environment_refractivity - refractivity) / (Environment_refractivity + refractivity),2);
    Reflectance_Scale = Reflectance_Scale + (1 - Reflectance_Scale) * pow((1 - coszetta), 5);//反射比例
    bool into = n.dot(r.d) < 0;//射入物体

    if (depth > min_deepth and dis(gen) > p) {
        return obj.e;
    }
    if (!into) {
        double temp;
        temp = Refractive_ratio_inv;
        Refractive_ratio_inv = Refractive_ratio;
        Refractive_ratio = temp;
    }
    if (transmission_probability > dis(gen) and pow(Refractive_ratio, 2) - pow(coszetta, 2) + 1 > 0) {//光线穿透

        Vec tra = r.d;//初始化折射方向
        int transmission_deep = 0;
        tra = Multiple_transmission(tra, n_Micro_surface, nl, bitangent, roughness, transmission_deep, Refractive_ratio_inv); //微表面多重折射
        f_tra = Vec_pow(f_tra, (transmission_deep/2.0));

        Vec ref = r.d;//初始化反射方向
        int reflect_deep = 0;
        ref = Multiple_reflections(ref, n_Micro_surface, nl, bitangent, roughness, reflect_deep); //微表面多重反射
        f_ref = Vec_pow(f_ref, reflect_deep);
        if (depth > min_deepth) {
            f_ref = f_ref * (1 / p);
            f_tra = f_tra * (1 / p);
        }
        Vec inv_n_Micro_surface = n_Micro_surface * (-1);
        if (ref == Vec()) {
            return obj.e +f_tra.mult(light_MIS(x, inv_n_Micro_surface, roughness)+radiance(Ray(x, tra), depth)) * (1 - Reflectance_Scale);
        }
        if (tra == Vec()) {
            return obj.e + f_ref.mult(light_MIS(x, n_Micro_surface, roughness) + radiance(Ray(x, ref), depth)) *  Reflectance_Scale;
        }
        if (depth < min_deepth) {
            return obj.e + f_ref.mult(radiance(Ray(x, ref), depth) + light_MIS(x, n_Micro_surface, roughness)) * Reflectance_Scale + f_tra.mult(light_MIS(x, inv_n_Micro_surface, roughness)+radiance(Ray(x, tra), depth)) * (1 - Reflectance_Scale);
        }
        else {
            if (dis(gen) < Reflectance_Scale) {
                return obj.e + f_ref.mult(radiance(Ray(x, ref), depth) + light_MIS(x, n_Micro_surface, roughness));
            }
            return obj.e + f_tra.mult(light_MIS(x, inv_n_Micro_surface, roughness)+radiance(Ray(x, tra), depth));
        }
    }
    Vec ref = r.d;//初始化反射方向
    int reflect_deep = 0;
    ref = Multiple_reflections(ref, n_Micro_surface, nl, bitangent, roughness, reflect_deep); //微表面多重反射
    if (ref == Vec()) {
        return obj.e;
    }
    f_ref = Vec_pow(f_ref, reflect_deep);
    if (depth > min_deepth) {
        f_ref = f_ref * (1 / p);
    }
    return obj.e + f_ref.mult(radiance(Ray(x, ref), depth) + light_MIS(x, n_Micro_surface, roughness));

}


// 存储窗口大小
int windowWidth = 2024;
int Width = windowWidth;
int windowHeight = 1080;
int Height = windowHeight;
int all = Width * Height;
std::vector<Vec> color(all);

int numThreads = omp_get_max_threads();
std::vector<double> maxcolor(numThreads);
std::vector<double> HDR_avg_color_lum(numThreads);

int ray_sum = 0;

int min_sample = 3,max_sample = 16;//采样次数
double Noise_Threshold = .05;//噪波阈值
Matrix M = Euler_Rotation_matrix(-.5, 0, 0);
Vec camara(0, 10, 20.5);
Vec lookat = (M.Matrix_dot(Vec(0, 0, -1))).norm();
Vec vup = M.Matrix_dot(Vec(0, 1, 0)).norm();
Vec right = (lookat % vup).norm();
double aspect_ratio = double(Width) / double(Height);
double inv_maxsample = 1.0 / double(max_sample);
double inv_minsample = 1.0 / double(min_sample);
double invw = 1.0 / double(Width), invh = 1.0 / double(Height);
double FOV_angle = 45;//视场角
double FOV_radian = FOV_angle * PI / 180;//弧度
double fov_chache = tan(FOV_radian * 0.5);//乘数缓存
double time_flash = 640 * min_sample;//
int ray_sum_chache = 0;//暂存已解算
int last_ray_sum = 0;//暂存已绘制
bool HDR_mode = 1;
bool frame_done = 0;
double Key = 0.18;//整体亮度（0.045~0.72）正常值0.18（18度灰）

inline int clamp(double& x,double a,double b) {
    x = (((x > a) ? a : x) < b) ? b : x;
    return 0;
}
inline void render()
{
   // glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);
    //HDR
    if (!frame_done) {

        double chache, oneover_Lscale_maxcolor_pow2;
        if (HDR_mode) {
            double maxmaxcolor = 0;
            double SUM_HDR_avg_color_lum = 1e-5;
            for (int c = 0; c < numThreads; c++) {
                SUM_HDR_avg_color_lum = SUM_HDR_avg_color_lum + HDR_avg_color_lum[c];
                if (maxmaxcolor < maxcolor[c]) {
                    maxmaxcolor = maxcolor[c];
                }
            }
            SUM_HDR_avg_color_lum = exp(SUM_HDR_avg_color_lum / double((ray_sum_chache==0)?1: ray_sum_chache));//平均颜色（基准）

            //if (1) { std::cout << HDR_avg_color_lum[0] << "error   \n"; }
            chache = Key / SUM_HDR_avg_color_lum;
            double Lscale_maxcolor = maxmaxcolor * chache;
            oneover_Lscale_maxcolor_pow2 = 1 / pow(Lscale_maxcolor, 2);
        }
    

        for (int i = 0; i < ray_sum_chache; i++) {
            //(HDR_mode)?0: last_ray_sum
            //HDR_计算缩放因子
            double Lscale_d=1;
            if (HDR_mode) {
                double Lscale = color_light(color[i]) * chache;
                Lscale_d = (Lscale * (1 + Lscale * oneover_Lscale_maxcolor_pow2)) / (1 + Lscale);//压缩高亮
                //clamp(Lscale_d,1e20,0);
            }
            //归一化
            // 执行绘制操作
            glColor3f(GLfloat((color[i]* Lscale_d).x), GLfloat((color[i] * Lscale_d).y), GLfloat((color[i] * Lscale_d).z));
            double dx = 2.0 * invw;
            double dy = -2.0 * invh;
            int x = int(fmod(i, Width));
            int y = i / Width;
            glVertex2f(GLfloat(dx * x - 1.0f), GLfloat(dy * y + 1.0f));  // 左下角顶点
            glVertex2f(GLfloat(dx * ++x - 1.0f), GLfloat(dy * y + 1.0f));   // 右下角顶点
            glVertex2f(GLfloat(dx * x - 1.0f), GLfloat(dy * ++y + 1.0f));    // 右上角顶点
            glVertex2f(GLfloat(dx * --x - 1.0f), GLfloat(dy * y + 1.0f));   // 左上角顶点
        }
        last_ray_sum = ray_sum_chache;

        std::cout << "正在渲染……" << double(last_ray_sum) / double(all)*100 << "%  \n";

    }
    glEnd();

    if (last_ray_sum >= all) {
        frame_done = 1;
    }
    glutSwapBuffers();
    //glFlush();
}
inline void ray_tracing(int value) {
    #pragma omp parallel num_threads(numThreads)
        {
            int threadID = omp_get_thread_num();
            int sum;
            sum = ray_sum + threadID;
            if(sum<all){
                Vec res, dt_sum_avg_res, last_sum_avg_res, sum_res;
                int i = sum / Width;
                int j = int(std::fmod(sum, int(Width)));
                int s;
                int flag = 0;
                for (s = 0; s < max_sample; s++) {
                    /*double dx = pow(dis(gen) * 2 - 1, 3) * 0.5 + 0.5;
                    double dy = pow(dis(gen) * 2 - 1, 3) * 0.5 + 0.5;*/
                    double dx = dis(gen);
                    double dy = dis(gen);
                    Vec dir2D = vup * ((i * invh - .5) * 2) * -1 + right * ((j * invw - .5) * 2) * aspect_ratio + right * dx * invw + vup * dy * -1 * invh;
                    Vec dir3D = lookat + dir2D * fov_chache;
                    last_sum_avg_res = sum_res * (1.0 / ((s < 1) ? 1 : s));
                    res = radiance(Ray(camara, dir3D.norm()), 0);
                    sum_res = sum_res + res;

                    dt_sum_avg_res = sum_res * (1.0 / (s + 1)) - last_sum_avg_res;
                    if (Vec_max(abs(sum_res * (1.0 / (s + 1)) - last_sum_avg_res)) < Noise_Threshold) {
                        flag++;
                    }
                    else {
                        flag--;
                    }
                    if (flag > min_sample) {
                        s++;//进行一个手动的自增
                        break;
                    }
                    last_sum_avg_res = sum_res * (1.0 / (s + 1));
                }
                sum_res = sum_res * (1.0 / double(s));
                //HDR
                HDR_avg_color_lum[threadID] += log(1 + color_light(sum_res));
                if (color_light(sum_res)> maxcolor[threadID]) {
                    maxcolor[threadID] = color_light(sum_res);
                }
                color[sum] = sum_res;
            }
        }
//#pragma omp barrier
    ray_sum += numThreads;
    glutTimerFunc(0, ray_tracing, 0);
}
// 窗口大小变化回调函数
inline void reshape(int width, int height)
{
    glViewport(0, 0, width, height);

    // 更新窗口大小
    windowWidth = width;
    windowHeight = height;
    frame_done = 0;
}

inline void update(int value)
{
    ray_sum_chache = (ray_sum > all) ? all: (ray_sum+1);//暂存
    render();
    //glutPostRedisplay(); // 通知窗口重绘
    glutTimerFunc(time_flash, update, value); // 设置下一次定时器触发
}


// 键盘事件回调函数
inline void keyboard(unsigned char key, int x, int y)
{
    if (key == 27) // ESC键(save)
    {
        FILE* f = fopen("image.ppm", "w");
        fprintf(f, "P3\n%d %d\n%d\n", Width, Height, 255);
        //HDR
        Vec maxmaxcolor(0, 0, 0);
        double SUM_HDR_avg_color_lum = 0;
        for (int c = 0; c < 16; c++) {
            SUM_HDR_avg_color_lum = SUM_HDR_avg_color_lum + HDR_avg_color_lum[c];
            if (color_light(maxmaxcolor) < maxcolor[c]) {
                maxmaxcolor = maxcolor[c];
            }
        }
        SUM_HDR_avg_color_lum = exp(SUM_HDR_avg_color_lum / double(all));//平均颜色（基准）

        double Lscale_maxcolor = color_light(maxmaxcolor) * (Key / SUM_HDR_avg_color_lum);
        double Lscale_maxcolor_pow2 = pow(Lscale_maxcolor, 2);
        for (int i = 0; i < all; i++) {
            //HDR_计算缩放因子
            double Lscale = color_light(color[i]) * (Key / SUM_HDR_avg_color_lum);
            double Lscale_d = (Lscale * (1 + (Lscale / Lscale_maxcolor_pow2))) / (1 + Lscale);//压缩高亮
            //double Lscale_d = (Lscale_d / (Lscale_d + 1));
            //归一化
            color[i] = color[i] * Lscale_d;

            color[i].x = (color[i].x > 1.0) ? 1 : color[i].x;
            color[i].y = (color[i].y > 1.0) ? 1 : color[i].y;
            color[i].z = (color[i].z > 1.0) ? 1 : color[i].z;
            fprintf(f, "%d %d %d ", int(color[i].x * 255.999), int(color[i].y * 255.999), int(color[i].z * 255.999));
        }
        exit(0);
    }
    else if (key == 72) {
        HDR_mode = !HDR_mode;
        frame_done = 0;
        std::cout << "HDR " << (HDR_mode ? "启用" : "禁用") << "\n";
    }
}


int main(int argc, char* argv[]) {
    for (int k = 1; k < 21; k++) {
        double size = dis(gen) * .5+.8;
        double R=0, G=0, B=0;
        hsv_to_rgb(int(dis(gen) * 360), 100, 100, &R, &G, &B);
        Vec RGB_ = Vec(R, G, B);
        //明度矫正
        double lineer = color_light(RGB_);
        RGB_ = RGB_ * lineer + Vec(1, 1, 1) * (1 - lineer);

        spheres[k]= Sphere(size, Vec(fmod(k-1,4)*6-9, size, (k-1)/4*6-15), Vec(), RGB_, dis(gen)*.05*lineer, pow(dis(gen),1), dis(gen)*.12+1.03);
    }
    for (int kk = 0; kk < 9; kk++) {

        double R = 0, G = 0, B = 0;
        hsv_to_rgb(int(dis(gen) * 360), 100, 100, &R, &G, &B);
        point_Lights[kk] = Point_Light(1, Vec(fmod(kk, 3) * 15 - 15, 10, kk / 3 * 12 - 15), Vec(R,G,B) * (dis(gen)*.2+.8) * 5000);
    }
    std::cout << "渲染分辨率" << windowWidth << "*" << windowHeight<<"\n";
    std::cout << "已启用线程x" << numThreads << "\n";
    std::cout << "最少采样x" << min_sample << " 最大采样x" << max_sample <<" 最大噪波阈值:"<< Noise_Threshold << "\n";
    std::cout << "HDR " << (HDR_mode?"启用":"禁用") << "\n";
    std::cout << "中性灰" << Key << "（HDR）" << "\n";
    std::cout << "FOV " << FOV_angle << "(角度)" << "\n";
    std::cout << "RR最小深度 " << min_deepth << "\n";
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("Path Tracing OpenGL Window");

    glutDisplayFunc(render);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutTimerFunc(0, ray_tracing, 0);
    glutTimerFunc(time_flash, update, 0);
    glutMainLoop();
    return 0;
}
