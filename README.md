# 环境搭建

**g-net**

Linux Kernel 4.15

ubuntu 20.0

dpdk 18.02

cuda 9.2

gcc 7.3 

Server 2

## 搭建问题

1. dpdk 18.02需要去官网手动下载

2. dpdk 18.02在编译过程中会存在各种各样的问题

   a. 需要在dpdk的GNUmakefile中指定如下命令

   KBUILD_CFLAGS += $(call cc-disable-warning,unused-but-set-variable)

   b. 需要在dpdk/x86_64-native-linuxapp-gcc/build/lib/librte_eal/linuxapp/igb_uio目录下的makefile中指定MODULE_CFLAGS += -Wno-implicit-fallthrough

   c. 在:~/Raven/g-net/dpdk/lib/librte_net$:149:2的指针强制转换造成的 may result in an unaligned pointer value [-Werror=address-of-packed-member]问题

   需要把rth_ether.h中的is_broadcast相关函数中判断广播的内容改成循环判断。

   **dpdk的编译依旧需要通过build_dpdk.sh来编译**

3. install.sh 脚本无法正确执行，因此大页面设置和网卡端口绑定需要手动执行 make_huagepage.sh

4. 默认的1024的内存页大小不够，需要4096 位于make_hugepage

5. 必须要在sudo下运行，否则无妨访问硬件

6. 缺失pkt-config

7. cannot get hugepage info

   linux下hugepage的设置

   查看/proc/meminfo 来得知hp的分配清空，跑当前目录下make_huagepage.sh

   通过-m命令去显示指定

8. apt-install失败，一般解决方法：修改sudo vi /etc/resolv.conf下的nameserver 8.8.8.8

9. 如果uio没有，那么通过dpdk-setup.sh去搞定

   如果shell脚本安装错误，执行如下命令

   ```shell
   [root@localhost dpdk-18.11]# modprobe uio
   [root@localhost dpdk-18.11]# cd build/kmod/
   [root@localhost kmod]# insmod igb_uio.ko
   [root@localhost kmod]# lsmod | grep uio
   ```

   上述完成直接运行绑定脚本即可

10. 必须要用sudo来跑dpdk demo

11. /usr/bin缺失文件 cannt find -lnids 说明缺乏库文件 nids 在包libnids中 这个是用来解析tcp信息包的

    sudo apt-get install libnet-dev

    sudo apt-get install libpcap-dev

    sudo apt-get install libnids-dev

    由于服务器无法连接上网络，因此只能手动安装。

12. onvm中全局使用的onvm_common.h time_diff函数需要是staic inline的

13. EAL: Can only reserve 275 pages from 4096 requested
    Current CONFIG_RTE_MAX_MEMSEG=256 is not enough
    Please either increase it or request less amount of memory.

    页面不够的问题，通过修改dpdk/x86.../.config下的CONFIG_RTE_MAX_MEMSEG来解决。我目前服务器上的大小是512

    **需要注意修改后dpdk,onvm都需要重新编译**

14. **先编译pstack再编译onvm**

15. 运行过程中告知cpu核心数不够

    查看核心数命令lscpu

    如下运行

    ```shell
    bash ./go.sh [numofcpu]
    ```

在当前server上至少要15个核心

16. 出现make nothing to be done for xxx 在编译onvm时，先make clean，再make

## 整个环境

自行下载dpdk 18.02和g-net，修改dpdk内makefile，使用build_dpdk.sh进行首轮编译后，进入x86文件夹修改头文件，makefile以及config，再进行编译。运行make_hugepages.sh以级dpdk-setup.sh配置大页面，uio和硬件绑定（必须绑定INTEL X系列网卡），再sudo运行看看demo情况。再编译onvm前，先编译pstack，再修改onvm头文件中的函数前缀，最后再编译onvm。

**下载g-net和dpdk前需要先将环境变量配置完毕，类似openNetVM**

# 源码笔记

## 项目启动

在onvm下启动go.sh 只需要给入一个参数 表示运行CPU个数

默认至少cpu个数要大于等于15个。从0开始的偶数序列。

nf启动参数

bash ./go.sh "1 3 5" 2

需要注意除了主线程最多个cpu外，最多8个cpu(因为batch_size最大就只为0)，并且由于线程数和核心数严格对于，**线程数必须等于除了主线程外的核心数**

### **系统全局变量**

### **系统的启动链**

调init，初始化端口号数组，service_chain，clients，nf_info，manager等后，启动五大核心线程。

### **NF的启动链**

首先启动nf本身模块，即onvm_nflib_init，将其内容加入nf_info_ring（**这一步主要分配各项内存client,nf_info等，并初始化info**），scheduler中调用check_status，检查status为waitting_for_id的nf，并调用onvm_nf_start(**该函数主要初始化client相关信息，设置服务链信息以及将nf串起来**)，最后启动manager_nf_init(**该函数主要用来初始化gpu相关信息**)。上述这些内容的初始化都是通过manager来完成的。

nf本身回进入framework_cpu循环和framework_gpu循环来进行处理。

### **数据流链**

与数据流相关的全局数据结构

```c
//定义在manager.c中
nf_request_queue
nf_response_pool
nf_response_pool

//定义在init中
clients
nf_info
service_chain
```

## NFV开发

一共有4个NF模块

1. ipsec
2. firewall
3. netgate
4. router

每个NF模块的开发流程基本上是类似的 **这里以gpu执行模块下的为例**

1. **加载gpu模块**

   onvm_nflib_init(argc,argc,NF_TAG,NF_NUMBER,GPU_NF,&(void gpu模块加载函数))返回>0正常，否则退出程序。并且argc-=offest argv+=offset

   gpu模块加载函数里同时需要引入.ptx(nvcc中间文件)和.cu文件

   **在cpu运行模块中需要加入判段标志位参数**

2. **设置内存大小并且分配全局变量**

   init_main(void) 通过gcudaAllocSize 先**开辟**每个线程需要处理的内存空间大小，包括输入/输出的CUDAdeviceptr，返回的int类型大小，以及全局变量大小，**注意要乘上BATCH_SIZE**。再通过gcudaMalloc和hostMalloc为全局变量**分配**内存，并将全局变量通过gcudaMemcpyHtoD拷贝给gpu。

   **注意开辟和分配的区别**

3. **为每个线程数据包的host与device分配内存** gcudaMalloc gcudaHostAlloc

4. **预处理设置host输入 ** 得到描述符中的CUdeviceptr。

5. **将host输入拷贝到gpu输入**

6. **设置gpu kernel函数参数** 设置完成后，加载kernel

   第一位是参数个数，其余位数要和kernel形参对应，要偏移量来获取参数

7. **将GPU数据拷贝给host**

8. **后处理来得到端口信息** 用来同步gpu,cpu信息。

**上述2-8都是static静态函数来处理的**

通过 onvm_nflib_init(*argc*, *argv*, NF_TAG, NF_NAME,TAG ,NULL,&(gpu_install))

通过onvm_framework_start_cpu(&(init),&(pre),&(pos),&(cpu),TAG) 来设置cpu流程 init_host_buf,user_batch_func,user_post_func，cpu_handle TAG(GPU_NF|CPU_NF)

通过onvm_framework_start_gpu(同上) 来设置gpu流程 htod dtoh setarg

上述过程契合论文的pre_process htod setage dtoh post_process

## NFV例子

下面给出一个gpu仅仅只作掩码与运算的demo例子。

## 各模块分析

### Manager

作用：

起到NF与openNetVM和GPU之间中间层的作用。具体如下：

1. 为每个nf进行初始化
2. gpu虚拟化，共享上下文
3. 调度器
4. 与openNetVM core通信（并通过其来switch）
5. nf,gpu,openNetVM数据传输

#### **manager** 

相当于buffer proxy，作为数据流代理。数据处理相关函数如下：

1. **init_manager**

   获取gpu设备，创建gpu上下文，并且设置上下文相关数据。初始化一个全局的nf_request_queue，用来表示nf作gpu相关请求的缓冲器。接着为每个client分配最大线程个数的cuda流和四个监控事件(kern_start,kern_end,gpu_start,gpu_end),进行nf_response_queue的初始化，并且作global_response_q的初始化。并设置每个client的gpu信息。

   初始化request和response的存储空间(nf_request_pool,nf_response_pool)，并初始化gpu共享内存中的指针(**cuMemAlloc(&gpu_pkts_buf, GPU_BUF_SIZE * GPU_MAX_PKT_LEN + TX_GPU_BUF_SIZE)**)。gpu_pkts_tail = gpu_pkts_head = gpu_pkts_buf; (gpointer)

   最后，创建cuda stream(非阻塞)用来为manager层的rx线程作htod用。

2. **manager_thread_main**

   gpu状态机，用来处理nf gpu响应。在五大类线程运行中，是首个被拉起来的线程。各个状态如下表示。在进入主循环后，从nf_request_queue中得到首个req，并根据请求的标志信息去进入状态机。每一个cuda操作完成后，都会执行对应的host函数（cuStreamAddCallback）。

   a. **HOST_MALLOC** 从dpdk层分配内存空间 用在init_main的gcudaAllocSize(全局空间，以及host层的输入输出大小)

   b. **GPU_MALLOC** 从gpu内存中分配空间，只分配，不初始化！

   c. **GPU_MEM_HtoD_ASYNC/SYNC** 值从host复制到device

   d. **GPU_MEM_DtoH_ASYNC/SYNC** 从device复制到host

   e. **REQ_GPU_LAUNCH_STREAM_ASYNC|ALL** launch kernel 异步调用会回调kernel_callback，来把kernel的响应给nf

   f. **REQ_GPU_SYNC** 同步所有的gpu流，调用sync_callback

   g. **REQ_GPU_SYNC_STREAM** 调用stream_sync_callback

   h. REQ_GPU_RECORD_START

   i. REQ_GPU_MEMFREE

   j. REQ_GPU_MEMSET

   **f,g有啥区别，没懂**

3. **manager_nf_init** 用来初始化每个nf的gpu相关数据

#### **onvm_stats** 

中显示当前onvm数据处理情况信息，在scheduler线程中运行。

1. **onvm_stats_display_all**
2. **onvm_stats_display_ports** 

#### **onvm_init** 

系统全局启动的函数以及全局变量的定义，包括port,nf client等。

```c++
struct client *clients = NULL; //保存了所有nf信息的数组
struct port_info *ports = NULL; //保存了所有ether信息的数组

//接受数据缓冲区，需要注意，g-net中，所有数据都使用一个缓冲区
struct rte_mempool *pktmbuf_pool;
struct rte_mempool *nf_info_pool;
pstack_thread_info pstack_info;

struct rte_ring *nf_info_queue;

uint16_t **services;
uint16_t *nf_per_service_count;
struct onvm_service_chain *default_chain; //服务链
struct onvm_service_chain **default_sc_p;
```

**系统中涉及到的所有环形缓冲区**

*struct* rte_ring \*nf_request_queue; 请求用环形缓冲区

**核心函数如下**

1. **init(argc,argv)** 初始化全局变量，启动manager，并创建gpu上下文，具体如下

   首先创建dpdk eal,初始化以太网端口信息，解析参数，各项初始化顺序如下

   ```c++
   ports = rte_malloc(MZ_PORT_INFO, sizeof(*ports), 0);//开辟端口空间，
   init_mbuf_pools();//初始化pktmbuf_pool
   init_client_info_pool();//初始化nf_info_pool
   init_port();//初始化端口信息，比如转发队列之类的，ports->tx_qs就是在其中初始化的，对每个端口的两个nf：tx_qs进行初始化,第一维是portId,第二维是nf_queue_id
   init_shm_rings();//初始化client数组，用mz,同时，初始化服务信息数据
   init_info_queue();//初始化nf_info_queue
   init_manager();//获取gpu上下文，设置对应参数，初始化每个client的response（包括了global的和非global的），初始化nf_request_pool,nf_response_pool,nf_request_ring
   init_pstack_info_pool();//初始化pstack??? 干什么用的
   ```

   接下来初始化**服务链**

   使用onvm_sc_create创建服务链(default_chain)，以服务链最大长度为循环体去循环服务链，每次通过onvm_sc_append_entry(default_chain, ONVM_NF_ACTION_TONF, service_chain[i])去在服务链添加service。

   **需要注意service_chain是固定数组**，其中存储了每个service的id。在rx_thread_main中会得到首个service id的nf作为起始项。如果启动顺序和service_chain顺序不一致。那么在onvm_nf_start函数中，更新nf_per_service_count[nf_info->service_id]]时，更新的可能不是第一个服务链的数据，使得rx_thread_main中始终无法更新得到第一个nf的rx_qs，从而无法接受数据。

2. 

#### **onvm_nf** 

nf功能的加载

1. **onvm_nf_start** nf启动模块，在这个模块中会分配instance_id(**注意service_id和instance_id不是一个东西**，service id只是service标识，instance_id是nf在clients中的下标)。更新client[instance_id]，nf_per_service_count\[nf_info->service_id\](每个service_id的服务)。得到在服务链中的下标，根据该下标确定新建的nf是在服务链头，链尾还是链中。

   如果在链中，那么前一个nf的tx_q_new就是新一个nf的rx_q_new,当前项的tx是后一项的rx。（因此framework中的传值才能成立）。如果是链尾，那么tx直接设置尾port的tx，也就是直接进入manager的tx转发队列。如果是链头，那么就不设置tx,rx的指向关系。

   上述完成后，更新nf_info信息，进入manager_init()，设置gpu相关信息。

   **nf_info相关信息已经在onvm_nflib_init中设置了**,onvm_nf_start是由scheduler调用的。

#### **main**

整个系统的启动，分配核心数给rx,tx,manager,调度器,gpu，至少需要的核心数如下

rx_lcores+tx_lcores+3

分别启动：manager状态机，tx线程，rx线程，gpu线程，main主线程用来作为调度器。

通过rte_get_next_lcore得到当前核心数，并通过rte_eal_remote_launch在指定核心上运行线程。例子如下:

```c++
cur_lcore = rte_get_next_lcore(cur_lcore, 1, 1);
if (rte_eal_remote_launch(manager_thread_main, NULL, cur_lcore) == -EBUSY) 
{
		RTE_LOG(ERR, APP, 
                "Core %d is already busy, can't use for Manager\n"
                cur_lcore);
		return -1;    
}
```

接受数据与转发数据涉及的数据结构，按批接受数据。

```c++
typedef struct rx_batch_s {
    volatile int gpu_sync __rte_cache_aligned;
    volatile int full[RX_NUM_THREADS] __rte_cache_aligned;
	//数据包在gpu 共享内存中的位置 gpointer
    CUdeviceptr buf_head;
	//从dpdk中拿到的数据包个数数组
    unsigned pkt_cnt[RX_NUM_THREADS];
	//从dpdk拿到的数据包,rx每一个执行线程，会有一个缓冲区
    struct rte_mbuf *pkt_ptr[RX_NUM_THREADS][RX_BUF_PKT_MAX_NUM];
    uint8_t buf[RX_NUM_THREADS][RX_BUF_SIZE];
#ifdef MEASURE_RX_LATENCY
    struct timespec batch_start_time[RX_NUM_THREADS];
#endif
} rx_batch_t;
```

根据核心线程去分析

1. **manager_thread_main**

   该主线程主要从nf_request_queue中获得gpu调用请求/物理内存管理请求，并且执行相对的状态机函数。

2. **rx_thread_main**

   初始化单次处理的批，rx线程信息，核心id等等。

   主循环下循环所有的端口号，对于每一个端口号，通过队列id以及端口号用dpdk等到数据包。

   遍历所有得到的数据包，判断其是否超过batch大小。如果没有超过，那么在rx_batch[batch_id].pkt_ptr\[thread_id][batch_cnt++]中存储得到的数据包。**这里得到的其实不是数据包本身，而得到的是数据包的描述符rte_mbuf**

   结构如下:

   ```c++
   void* buf_addr;//之所以可以直接将rte_mbuf直接转换，也是因为因为其第一项是指向实际负载的指针。
   ....
   int hash;
   ```

   更新用户层数据的指针位置为pos（当前数据位置）。将得到数据包描述符其hash指针强制转换为用户层数据的Cuda格式指针(CUdeviceptr)，从而作为**gpointer**，根据pos在用户层空间的偏移量，来得到当前数据包在gpu层的位置。并将数据包内的数据转换成**gpu_packet_t**格式。

   **rx接受数据是根据批来处理的，每个批又有四个线程缓冲区处理数据**

   上述操作完后，pkt数据进入了rx_batch中的pkt_ptr(**dpdk格式**)以及buf中(**gpu_packet_t**)

3. **rx_gpu_thread_main**

   设置当前所使用的cuda上下文，设置单次处理信息

   在主循环中，先从第一个rx_batch开始处理数据，判断当前数据包有无超过gpu内存边界(环形缓冲区的体现)，得到当前数据包在gpu内存中的位置(**gpu_pkts_head**,与rx_thread_main中作为描述符传递不同，这里得到数据包位置是为了作h2d)，并更新gpu_pkts_head使其指向下一个将要来到的数据包，最后更新batch_id。

   得到第一个nf服务的rx_qs(接受数据队列)，将当前批所有的dpdk层数据全部传递给rx_qs。（**由此，第一个nf就得到了dpdk格式的数据**）

   每完成一个批处理，都会把当前批的gpu_packet_t格式的数据传递给gpu_head指定的gpu内存位置。

   **上述过程完成后，rx_batch的数据就被存储到了第一个nf的rx_qs以及其对应gpu内存中**

​	**在数据被传递到manager层后，随机就在三个地方保存了副本，manager中的批，gpu内存，nf的rx_qs队列中。**

​	host层存储的数据和gpu存储的数据不是一个格式，gpu层存储的数据采用了gpu_packet_s格式,cpu层存储的数据采用了rte_mbuf格式(原生格式)

4. **tx_thread_main**
5. **schedule_thread_main** 该线程执行三个工作，display系统信息，初始化nf，gpu调度。

一共三种调度方式，调度统计，静态/动态调度

### nflib

基本库

#### **onvm_nfilb**

onvm 基本api

头文件相关定义

具体函数

1. **onvm_nflib_init** 以secondary模式启动一个新的nf，并加载对应的gpu模块

   1. 从go.sh中解析得到该nf所需要用的参数
   2. 得到nf_info内存池 （通过rte_mempool_lookup得到内存池）
   3. 从内存池中分配内存给nf_info (使用rte_mempool_get)，并且使用onvm_nflib_info_init去初始化
   4. 得到数据包内存池
   5. 得到服务链内存池
   6. 得到nf_info队列
   7. 将nf传递给nf_info队列 （如果失败，释放内存使用rte_mempool_put）
   8. 检查nf_info信息，决定是否释放分配的内存
   9. 得到对进程client内存池并且指定client
   10. 获取cuda gpu模块的信息
   11. 得到response内存池，request内存池，request队列

   **上述过程处理nf_info外，均只分配了内存，没初始化**

   上述得到的这些量都会通过级联编译，从而在onvm_framework中通过extern得到。

2. onvm_nflib_stop() 关闭当前nf

3. onvm_nflib_handle_signal int信号(ctrl+c)用来关闭nf主循环

#### **onvm_common** 

系统关键结构体的声明，比如client portinfo等

1. onvm_service_chain 服务链信息

2. onvm_nf_info nf基本信息结构体，实例id,服务id,状态标志

3. client nf客户端信息

4. nf_req gpu调度的请求信息

   ```c++
   struct nf_req {
   	volatile uint16_t type;
   	volatile uint16_t instance_id; 
   	volatile uint16_t thread_id;
   	volatile CUdeviceptr device_ptr;
   	volatile uint32_t host_offset;
   	volatile uint32_t size;
   	volatile uint32_t value;
   	struct nf_req *next;
   };
   ```

5. nf_rsp gpu调度的响应信息

   ```c++
   struct nf_rsp {
   	volatile int type;
   	volatile int states;
   	volatile int batch_size;
   	volatile int instance_id;
   	CUdeviceptr dev_ptr;
   };
   ```

6. 各个nf的标志位

7. 各个数据包执行操作的标志位

#### **onvm_framework**

头文件定义的相关数据结构

1. **nfv_batch_s** 每个cpu处理模块都需要定义的一个模块
2. **context_s/t** 上下文id

相当于介于manager和nf之间的一个中间层，用来接受和指定各个nf的cpu执行步骤和gpu执行步骤，并且通过manager的状态机去具体执行指定的gpu步骤。

1. **onvm_framework_start_cpu** 每个nf的cpu相关所需要处理的函数 包括init_host_buffer(这个函数中，所有数据包会在host和device中各留下一份备份),user_batch_func(获取数据存储在host内存中),user_post_func(获取数据存储在host的内存)

   **启动cpu模块的调用链如下** start_cpu->spawn_thread->cpu_thread->thread_init->framework_cpu|cpu_only

   

2. **onvm_framework_start_gpu** nfv gpu处理的核心模块函数

   进入gpu主循环前必须先确保调用了cpu_thread

   1. 首先根据gpu线程id得到data_buf
   2. 在确保数据缓冲区并未被处理过，并且轮询cuda stream响应存在后，将device数据给host (**d2h**)，注意这里的host_res往往就不是CUdeviceptr类型的了。如果数据已经被处理过就进入cpu状态，等待拷贝给dpdk mem。
   3. 在batch数据包处理gpu_ready的情况下，得到其batch id
   4. 如果所需线程数比当前执行线程数多，那么就通过spawn cpu开辟更多线程。
   5. record gpu start
   6. 将数据拷贝给device (**h2d**)
   7. 设置cuda参数 (**setArgs**)
   8. 设标志位，启动kernel

   

   **解析参数setArgs**

   ​	gpu_schedule_info用来存储gpu所需要使用的参数和调度信息，在manager_init中，每个client会被分配一个由memzone开辟的gpu_schedule_info信息。在onvm_framework_init中，得到client的gpu_info，并且初始化线程数等信息。在user_gpu_set_arg中，**由arg_info来记录每个gpu参数的偏移量，args字节数组来记录每个参数实际值**，在该步中，会把device的CUdeviceptr传递给偏移后的args。

   

   ​	在最终的manager.c加载kernel阶段中，根据arg_info记录的偏移量，将args中的数据倒入到arg_info中，**将字节数组中存储的每个参数的首字节地址传递给arg_info**，并传递给cuLaunchKernel。（这个函数的args只接受类型为CUdeviceptr*的void**数组）

   

   ​	**gpu_schedule_info结构中args,arg_info也是二维结构的，第一维是默认的8个线程，第二维是每个线程实际使用的gpu 参数。因此每一个线程操作时，都取其中一个维度的args,arg_info进行。**

   

4. **cpu_thread** nfv中cpu处理的核心模块函数

   framework_cpu是cpu模块的核心函数，具体内容如下：

   在进入主循环前，先onvm_framework_thread_init中启动**init_func**初始化user空间和device空间内存

   1. 通过线程id得到当前batch_set中的batch
   2. 判断其是否进入了cpu_Ready状态
   3. 将batch数中的gpu描述符信息传递给当前项的pkt(**post_process**)，**这里的post_process用来处理cpu-gpu数据同步**
   4. 把dpdk mem中缓冲区进行重排序，将要drop的项整理到缓冲区后面
   5. 压入转发队列（tx），这一步应该就明确压入了client的tx_qs中。
   6. 释放drop项内存，并更新client
   7. 尽可能去从rx接受数据
   8. 将接受到的数据从dpdk mem传递给user buf（**pre_process**），**这里得到的数据是描述符，而不是数据本身**。在当前实现中，pre_process的主要功能是得到描述符中的gpu指针。
   9. 更新client信息，并将**当前buf信息设置为gpu状态**

   **实际上nf在启动的过程中有三层内存host(vm),device(gpu),dpdk(本机)**

   

5. **gcudaAllocSize** 调用master中的host alloc模块，仅仅起到得到host层memzone的作用

6. **gcudaMalloc** 调用master中的gpu malloc模块，通过cuMemAlloc()来根据CUdeviceptr来在gpu内存空间中分配大小

7. **gcudaHostAlloc** 为host alloc中分配的host空间进行初始化。

**NF运行的线程结构**

默认每个nf有8个线程， 每个线程处理一个nf_batch_s结构的batch，每个batch包含大小为3的缓冲区，每个缓冲区处理4096个数据。

**数据流相关数据结构**

```C++
//每个线程对应的批
typedef struct nfv_batch_s
{
    //批中的缓冲区
	void *user_bufs[NUM_BATCH_BUF];
	struct rte_mbuf **pkt_ptr[NUM_BATCH_BUF];

	int buf_size[NUM_BATCH_BUF];
	volatile int buf_state[NUM_BATCH_BUF];

	int thread_id;
	volatile int gpu_buf_id;
	int gpu_next_buf_id;
	int gpu_state;

	int queue_id;

	void *host_mem_addr_base;
	void *host_mem_addr_cur;
	int host_mem_size_total;
	int host_mem_size_left;
} nfv_batch_t;

//缓冲区中每个项的数据结构
typedef struct my_buf_s {
	/* Stores real data */
	CUdeviceptr *host_in;
	uint8_t *host_out;
	CUdeviceptr device_in;
	CUdeviceptr device_out;
} buf_t;
```

**数据流流向**

1. 初始化 在init_main中开辟一个线程所用的所有空间，包括3个数据缓冲区需要的i/o结构和全局数据结构，用memzone_reverse分配。在启动cpu线程主循环前，先使用thread_init去初始化三个数据缓冲区，包括了device_int/out,host_in/out(**gcudaMalloc&gcudaHostAlloc**)，这里使用的host空间会使用之前在init_main中在memzone分配的空间，使用host_mem_addr_base，host_mem_add_cur来记录空间下标。使用的device空间会额外分配。

   **虽然所有的数据包都使用了一个大块的gpu shared mem，但是每个nf依旧需要自己开辟一个device mem 用来存储数据的CUdeviceptr**

2. 数据获取 从cpu_thread中得到描述符包后，将描述符包的每一个元素中gpointer转换给目前正在处理的batch。每处理完一个batch，设置gpu_ready。

3. 拷贝 将得到的batch其host_in中的CUdeviceptr拷贝给device_in，从而获得gpu mem数据指针。接着配置参数，调用kernel。

   **cpu gpu间依旧会进行CUdeviceptr的数据传输**

#### **onvm_sc_mgr**

用来获取和创建服务链的函数，服务链作为一个全局变量，保管了每个nf对数据包的处理信息和数据包的方向信息。

1. **onvm_sc_get**
2. **onvm_sc_create ** 使用rte_calloc来创建default_chain

#### **onvm_sc_common**

用来创建和管理服务链模块

1. **onvm_sc_append_entry** 传入一个服务链，nf行为以及目的地，向服务链添加一个nf服务情况。会在这个函数中，设置服务链添加项的action和dst。

   ```C++
   chain->sc[chain_length].action = action;
   chain->sc[chain_length].destination = destination;
   ```

## CPU计算模块

思路，gpu处理模块由于cuda需要的数据模式规定，需要把dpdk数据和全局变量都转换成user层数据再变成cuda能用的格式。而对于CPU模块来说，我们可以直接在NF进程的内存空间里面开我需要的全局数据，直接用dpdk格式的数据包，处理完后再直接导入到gpu中。

### 宏观思路

1. 在onvm_framework中，给出计算模块标志位，头文件中新定义一个参数只为struct rte_mbuf*的函数指针，修改onvm_framework_start_cpu参数，添加上述两个。添加一个函数onvm_cpu_thread_only，这个函数中只处理cpu计算相关内存。添加一个cpu_batch_func全局变量。还需要一个extern int nf_handle_tag来表示当前nf类型。cpu_thread中，根据标志位会选择要启动的函数是cpu还是cpu_only。
2. 在onvm_nflib中，给出nf_handle_tag的定义。并且为onvm_nflib_init添加一个用来标志handle_type的参数，使得在gpu模式下不会调用init_gpu
3. 需要将onvm_framework中，所有涉及到gpu_info的信息进行检查，确保其只在GPU_NF的模式下运行

**上面这个做法有问题 会提示gpu 信息 client得不到，问题是出在client的gpu_info上面，需要在manager.c中进行修改，把和gpu相关的操作放置到gpu存在时才运行**

### 细节

​	需要注意：

1. 主线程原本在gpu模块中，会运行gpu相关的内容，目前仅在cpu上操作的化，主线程就没有其他工作了，**最好用信号机制去写一个处理keep_running的函数**。因此在framework中，就需要添加一个额外的onvm_cpu_wati函数，用siganl和onvm_nf_stop,以及循环阻塞来阻塞主线程。
2. 当我给每一个nf开多个线程，拉线程的时候会出现段错误。这是因为第一个之后的nf线程在thread_init中会使用gcudaAllocSize，而这个函数需要第一次调用后才能二次调用，原本我并没有在第一个线程加载时调用这个函数，最终导致之后线程也开不出来。

**在该系统中，一定一定一定，使用remote_launch时，一个cpu对应一个线程**

3. 如何通过目前接口 gcudaAllocSize等来获取数据给cpu模块使用。每个线程根据其线程号得到batch，每个batch间并没有交互。并且cpu模块只用dpdk层数据，而gcudaAlloc组都是分配用户层或者cuda层数据。因此直接舍弃用gcudaAlloc相关结构来分配内存，对于全局变量，nf的代码中直接静态变量，对于数据包，直接用从dpdk中拿数据就可以了。

4. 如何优雅的使用ctrl+c退出nf

   对于cpu-gpu流程的nf，在gpu流程中，通过SIGINT触发onvm_handle_...来置keep_running从而使得程序循环全部可以退出。

   对于单cpu流程的nf，自定义了一个onvm_framework_cpu_only_wait，信号触发，循环阻塞主线程，阻塞失效后，再stop_nf。

nf_handle_tag 位于onvm_nflib

CPU_NF,GPU_NF 位于onvm_framework

### 开发流程 

### CPU模块问题

1. 在CPU模块后调用GPU模块，会出现问题。接受数据时会出现问题。如果不接数据的话，是没有问题的。	

   这说明，系统在起nf的过程中是正确的，但是在纯cpu运算，到gpu模块运算间的数据交互是有问题的。

   问题定位在main:433行

   ```
   checkCudaErrors( cuMemcpyHtoDAsync(batch->buf_head, (void *)batch->buf, sizeof(batch->buf), stream) );
   ```

   中batch->buf_head非法，该问题还会导致cpu kernel函数调用时，报700错误。

   

## 数据包调度优化

通过client中的标志位(CR,CW,GR,GW)，每个标志位都是32位数组，每一位都表示了一个数据包信息被修改与否，让manager层根据每个client的标志位来得出所有NF的同步计划和最后DtoH的同步计划。

具体算法：每个NF的同步计划为上一项NF的修改计划(dirty vector)与当前项的读取计划(RV)，来得出当前NF有哪些项是需要CPU|GPU同步的，并依次迭代，得到最后从GPU到CPU有哪些项是需要同步的。

**问题：** 每加入一个NF就必须重新计算一次全局plan。

**实现细节：**

1. client中给出GR,GW,CR,CW,在init_she_mem中初始化最长client时，同时初始化client中的读写标志位。

2. 每次有一个nf入服务链，会在nf_start中，初始化该nf实际的读写标志位。

3. 每次一个nf加入服务链，都会在scheduler中，启动data synchronzation plan genertaion算法。

4. 在nf的数据处理过程中，根据plan数组中对应client的plan，在pro_process中进行CPUtoGPU的同步，在post_process中进行GPUtoCPU的同步。

5. plan数组作为全局变量，每个元素的结构如下

   ```C++
   
   ```

   

## 全局问题

5. 调度器和各个组件通信的问题

6. **每个nf只能开辟一个接受队列，否则会出现问题！！！**

   

## 系统测试

### benchmark

benchmark是测试模块

应该是会模拟数据发送给启动了g-net的另一个服务器。需要修改main.c中PKTLEN的大小，以及需要修改命令行参数如下：

```c++
-l 1 -n 4 --proc-type=primary
```

**在新的服务器上部署benchmark需要重新make dpdk** 而且要绑定到和onvm的同一个端口上

**要在root模式下启动**

### 测试指标

1. 接受吞吐量
2. 转发吞吐量


