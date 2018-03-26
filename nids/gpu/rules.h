#include <stdint.h>

#define MAX_PORTS 65536
#define MAX_STATE 100000
#define SYM_NUM 7

#define ALERT "alert"
#define TCP "tcp"
#define UDP "udp"
#define ICMP "icmp"

#define MSG 0x0
#define CONTENT 0x20000
#define DEPTH 0x01
#define OFFSET 0x02
#define WITHIN 0x04
#define DISTANCE 0x08
#define NOCASE 0x010// 0 for case sensitivity; 1 for case insensitivity.
#define RAWBYTES 0x020// 0 for not using raw bytes; 1 for using raw bytes;
#define HTTP_CLIENT_BODY 0x040
#define HTTP_COOKIE 0x080
#define HTTP_RAW_COOKIE 0x100
#define HTTP_HEADER 0x200
#define HTTP_RAW_HEADER 0x400
#define HTTP_METHOD 0x800// 1 for the search to the extracted Method from a HTTP client request
#define HTTP_URI 0x1000
#define HTTP_RAW_URI 0x2000
#define HTTP_STAT_CODE 0x4000
#define HTTP_STAT_MESSAGE 0x8000
#define FAST_PATTERN 0x10000

typedef struct _acNode
{
	char *str;
	int contId;
	struct _acNode *chdNode;
	struct _acNode *broNode;
	struct _acNode *failNode;
	int pattId;
	int root;
	int nodeNum;
} AcNode;

typedef struct _acNodeGPU
{
	int contId;
	int chdNode;
	int broNode;
	int failNode;
	int pattId;
	int root;
} AcNodeGPU;

typedef struct _acQueue
{
	AcNode *ac;
	struct _acQueue *next;
} AcQueue;

typedef struct _OptFpList
{
	char *context;

	int index; // contIndex

	uint8_t depth;
	uint8_t offset;
	uint8_t distance;
	uint8_t within;

	uint16_t flags;
	struct _OptFpList *next;

} OptFpList;

typedef struct _IpAddrSet
{
	uint32_t ip;
} IpAddrSet;

typedef struct _RuleSetRoot
{
	uint16_t acArray[MAX_STATE][256];
	int16_t failure[MAX_STATE];
	uint16_t acGPU[MAX_STATE * 257];
	//uint16_t **cpGPU;
	//uint16_t pattNum;
	AcNode *contPattMatch;
	AcNodeGPU *contPattGPU;
	uint16_t nodeNum;
} RuleSetRoot;

typedef struct _OptTreeNode
{
	OptFpList *opt_func;

	int type;
	int evalIndex; // where this value sits in the evaluation sets

	char *msg;

	// stuff for dynamic rules activation/deactivation
	/*int active_flag;
	  int activation_counter;
	  int countdown;
	  int activates;
	  int activated_by;

	  struct _RuleTreeNode *RTN_activation_ptr;
	  struct _OptTreeNode *OTN_activation_ptr;*/

	struct _OptTreeNode *next;
	struct _RuleTreeNode *rtn;
} OptTreeNode;

typedef struct _RuleTreeNode
{
	int type;
	IpAddrSet *sip;
	IpAddrSet *dip;
	uint16_t hdp[MAX_PORTS]; // 16bits
	uint16_t ldp[MAX_PORTS];
	int flags; // 32bits

	struct _RuleTreeNode *right;
	struct _OptTreeNode *down;
} RuleTreeNode;

typedef struct _RuleTreeRoot
{
	RuleTreeNode *rtn;
	RuleSetRoot *rsr;
} RuleTreeRoot;

typedef struct _RuleListRoot
{
	RuleTreeRoot *prmSrcGroup[MAX_PORTS];
	RuleTreeRoot *prmDstGroup[MAX_PORTS];
	RuleTreeRoot *prmGeneric;
} RuleListRoot;

typedef struct _ListRoot
{
	RuleListRoot *IpListRoot;
	RuleListRoot *TcpListRoot;
	RuleListRoot *UdpListRoot;
	RuleListRoot *IcmpListRoot;
} ListRoot;

/*
typedef struct _ContentIndexArray
{
	OptTreeNode *pattern;
	int *relContent;
}ContentIndexArray;

typedef struct _PatternContentArray
{
	ContentIndexArray *indexArray;
	OptFpList **contentIndex;
}PatternContentArray;*/

typedef struct _TmpRuleHeader
{
	int type;
	IpAddrSet *sip;
	IpAddrSet *dip;
	uint16_t hdp; // 16bits
	uint16_t ldp;
	uint32_t flags; // 32bits
} TmpRuleHeader;

typedef struct _TmpRuleOption
{
	char *msg;
	char *context;
	uint8_t depth;
	uint8_t offset;
	uint8_t distance;
	uint8_t within;
	uint16_t flags;
} TmpRuleOption;

ListRoot *configrules(const char *filename);
void precreatearray(ListRoot *listroot);
