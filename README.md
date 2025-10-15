# GAN for ocean temperature reconstruction
# 参考文献：谷浩然，杨俊钢，崔伟，等. 基于生成对抗网络的南海三维温度场重构研究[J]. 海洋学报，2025，47(4)：86–99 doi:  10.12284/hyxb2025035
<span style="font-family: SimSun, 'Times New Roman'; font-size: 12pt;">
	
  本研究使用的海洋表层卫星遥感数据包括、海面温度（SST）、海面盐度（SSS）和海面风场（SSW）。海面高度异常数据来自哥白尼海洋环境观测服务中心（CMEMS）提供的高度计L4网格数据，该数据使用最优插值法融合了不同卫星的L3沿轨数据，数据时间分辨率为每天，空间分辨率为（1/4）°。海面盐度数据是由CMEMS提供的L4网格数据，时间分辨率为每天，空间分辨率为（1/8）°。海面温度数据来自遥感系统（REMSS）提供的微波和红外L4融合数据，时间分辨率是每天，空间分辨率为（1/4）°。所使用的海面风场数据包括纬向风速SSWU和经向风速SSWV，数据来自REMSS遥感系统研制并发布的交叉校准多平台L3海面风场融合数据集，该数据采用变分同化方法融合了多个海洋微波辐射计和散射计所采集的海面风场数据，时间分辨率是每6小时，空间分辨率为（1/4）°。
  海洋次表层温度场数据是CMEMS提供的海洋再分析数据GLORYS12V1的三维温度场数据。该数据的时间分辨率为日平均和月平均，空间分辨率（1/12）°×（1/12）°，垂向为0-4000m共50层，提供的海洋参数包含温度、盐度、海流、海面高度、混合层深度和海冰参数。GLORYS（Global Ocean Reanalysis and Simulations）是MyOcean框架下开发的一项全球海洋资料再分析系统，在加入同化数据的约束下使用较高分辨率的网格对全球海洋进行模拟。该产品的海洋模式为欧洲海洋模型中心NEMO Version3.1和耦合海冰模式LIM2。GLORYS12V1的同化方案选择降阶卡尔曼滤波算法，同时选择3D-var修正温盐误差，用于同化的观测数据来自CMEMS的高度计数据、海面温度，以及CORA数据库中的现场观测温度和盐度剖面数据和Argo浮标数据。本研究所使用的所有数据的时间范围为2013年1月1日至2018年12月31日。  
  为了更有效地构建模型并提高训练效率，需要按照一定的标准化方法统一多源遥感数据的量纲。采用了最大最小归一化方法对输入数据进行标准化。
  
  所提供的示例数据按照每天的表层遥感数据和次表层再分析数据划分，每日数据为三个维度[经度、纬度、通道数(前三个变量为表层遥感数据，之后的变量为不同深度层再分析温度数据)]。  
	 

SLA（数据地址：https://doi.org/10.48670/moi-00148）  
SST（数据地址：https://www.remss.com/measurements/sea-surface-temperature）  
SSW（数据地址：https://www.remss.com/measurements/ccmp/）  
再分析数据（数据地址：https://doi.org/10.48670/moi-00021）  
</span>

<span style="font-size:20px; font-weight:bold">生成器网络结构</span>

<img width="755" alt="生成器2 18" src="https://github.com/user-attachments/assets/f70af777-1481-4e64-ad85-9fea2950ee5e" />

<span style="font-size:20px; font-weight:bold">鉴别器网络结构</span>

<img width="1608" alt="鉴别器加入SSS" src="https://github.com/user-attachments/assets/624fe867-fe74-4020-8a9e-ccaa17ebb267" />

