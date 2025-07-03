# GAN-for-ocean-temperature-reconstruction
  数据集按照每天的表层遥感数据和次表层再分析数据划分，包三个维度（经度、纬度、通道数）。
  本研究使用的海洋表层卫星遥感数据包括、海面温度（SST）、海面盐度（SSS）和海面风场（SSW）。海面高度异常数据来自哥白尼海洋环境观测服务中心（CMEMS）提供的高度计L4网格数据，该数据使用最优插值法融合了不同卫星的L3沿轨数据，数据时间分辨率为每天，空间分辨率为（1/4）°。海面盐度数据是由CMEMS提供的L4网格数据，时间分辨率为每天，空间分辨率为（1/8）°。海面温度数据来自遥感系统（REMSS）提供的微波和红外L4融合数据，时间分辨率是每天，空间分辨率为（1/4）°。所使用的海面风场数据包括纬向风速SSWU和经向风速SSWV，数据来自REMSS遥感系统研制并发布的交叉校准多平台L3海面风场融合数据集，该数据采用变分同化方法融合了多个海洋微波辐射计和散射计所采集的海面风场数据，时间分辨率是每6小时，空间分辨率为（1/4）°。所使用的现场观测数据为Argo浮标温度剖面数据。
  海洋次表层温度场数据是CMEMS提供的海洋再分析数据GLORYS12V1的三维温度场数据。该数据的时间分辨率为日平均和月平均，空间分辨率（1/12）°×（1/12）°，垂向为0-4000m共50层，提供的海洋参数包含温度、盐度、海流、海面高度、混合层深度和海冰参数。GLORYS（Global Ocean Reanalysis and Simulations）是MyOcean框架下开发的一项全球海洋资料再分析系统，在加入同化数据的约束下使用较高分辨率的网格对全球海洋进行模拟。该产品的海洋模式为欧洲海洋模型中心NEMO Version3.1和耦合海冰模式LIM2。GLORYS12V1的同化方案选择降阶卡尔曼滤波算法，同时选择3D-var修正温盐误差，用于同化的观测数据来自CMEMS的高度计数据、海面温度，以及CORA数据库中的现场观测温度和盐度剖面数据和Argo浮标数据。本研究所使用的所有数据的时间范围为2013年1月1日至2018年12月31日。

  SLA（数据地址：https://doi.org/10.48670/moi-00148）
  SST（数据地址：https://www.remss.com/measurements/sea-surface-temperature）
  SSW（数据地址：https://www.remss.com/measurements/ccmp/）
  ARGO（数据地址：http://www.argodatamgt.org）
  再分析数据（数据地址：https://doi.org/10.48670/moi-00021）

<img width="755" alt="生成器2 18" src="https://github.com/user-attachments/assets/f70af777-1481-4e64-ad85-9fea2950ee5e" />
<img width="1608" alt="鉴别器加入SSS" src="https://github.com/user-attachments/assets/624fe867-fe74-4020-8a9e-ccaa17ebb267" />

