#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <regex.h>
#include <ctype.h>

#include "rules.h"

#define MAX_CH_BUF 10000

int contIndex = 1;
int newState = 0; // newState must be less than or equal to 2^16
int nodeNum = 0;
int pattNum = 0;

static void buildACarray(RuleSetRoot *rsr, OptFpList *fplist)
{
	// goto function
	int state = 0, i = 0;
	char ch;

	while (1) {
		ch = fplist->context[i];
		if (ch == '\0') {
			break;
		} else if ((rsr->acArray[state][(int)ch] & 0xffff) == 0) {
			break;
		} else {
			state = rsr->acArray[state][(int)ch];
			i++;
		}
	}

	if (ch != '\0') {
		while (ch != '\0') {
			newState++;
			rsr->acArray[state][(int)ch] = newState;
			state = newState;
			ch = fplist->context[++i];
		}
		rsr->acArray[state][(int)ch] = fplist->index; //<< 16;
	} else {
		fplist->index = rsr->acArray[state][(int)ch]; //(rsr->acArray[state][(int)ch] & 0xffff0000) >> 16;
	}
}

static void buildfailfunc(RuleSetRoot *rsr)
{
	int tmp_queue[MAX_STATE];
	int flag_queue = -1;
	int tail_queue = -1;
	int state = 0;
	int i;

	memset(tmp_queue, 0, MAX_STATE * sizeof(int)); 
	for (i = 0; i < 256; i++) {
		if (rsr->acArray[state][i] != 0) { 
			tail_queue++;
			tmp_queue[tail_queue] = rsr->acArray[state][i]; //& 0x0ffff;
		}
	}

	while (tail_queue != flag_queue) {
		flag_queue++;
		state = tmp_queue[flag_queue];
		for (i = 1; i < 256; i++) {
			if (rsr->acArray[state][i] != 0) {
				tail_queue++;
				tmp_queue[tail_queue] = rsr->acArray[state][i]; //& 0x0ffff;
				int tmp_state = 0;
				tmp_state = rsr->failure[state];
				while (rsr->acArray[tmp_state][i] == 0) {
					if (tmp_state == 0) break;
					tmp_state = rsr->failure[tmp_state];
				}

				rsr->failure[tmp_queue[tail_queue]] = rsr->acArray[tmp_state][i];
			}
		}
	}
}

static AcNode *buildContPattMatch(AcNode *acNode, OptFpList *fplist)
{
	int flag = 0;
	AcNode *tmp = acNode->chdNode;
	while (tmp != NULL) {
		flag = 1;
		if (tmp->contId == fplist->index) {
			acNode = tmp;
			return acNode;
		} else {
			tmp = tmp->broNode;
		}
	}

	nodeNum++;
	AcNode *tmp_acNode;
	tmp_acNode = (AcNode *)malloc(sizeof(AcNode));
	tmp_acNode->str = fplist->context;
	tmp_acNode->contId = fplist->index;
	tmp_acNode->chdNode = NULL;
	tmp_acNode->broNode = NULL;
	tmp_acNode->failNode = NULL;
	tmp_acNode->pattId = 0;
	tmp_acNode->root = -1;
	tmp_acNode->nodeNum = nodeNum;

	if (flag == 0) {
		acNode->chdNode = tmp_acNode;
		acNode = acNode->chdNode;
	} else {
		tmp = acNode->chdNode->broNode;
		acNode->chdNode->broNode = tmp_acNode;
		tmp_acNode->broNode = tmp;
		acNode = tmp_acNode;
	}
	return acNode;
}

static void buildfailContPattMatch(AcNode *acNode)
{
	AcNode *tmp_acNode;
	tmp_acNode = acNode;
	AcQueue *flag_ac = NULL;
	AcQueue *tail_ac = NULL;
	AcQueue *tmp_ac;
	AcQueue *tmp;

	tmp_acNode = acNode->chdNode;
	if (tmp_acNode != NULL) {
		tmp_acNode->failNode = acNode;
		flag_ac = (AcQueue *)malloc(sizeof(AcQueue));
		flag_ac->ac = tmp_acNode;
		flag_ac->next = NULL;
		tail_ac = flag_ac;
		tmp_acNode = tmp_acNode->broNode;
	}

	while (tmp_acNode != NULL) {
		tmp_acNode->failNode = acNode;
		tmp_ac = (AcQueue *)malloc(sizeof(AcQueue));
		tmp_ac->ac = tmp_acNode;
		tmp_ac->next = NULL;
		tail_ac->next = tmp_ac;
		tail_ac = tmp_ac;
		tmp_acNode = tmp_acNode->broNode;
	}

	while (flag_ac != NULL) { // flag_ac is the top of the AcQueue.
		tmp = flag_ac;
		tmp_acNode = tmp->ac->chdNode;
		while (tmp_acNode != NULL) {
			// add the original flag_ac(tmp->ac)'s child nodes and find the failure node of each child(tmp_acNode)
			// failure function
			AcNode *upper_fail = tmp->ac->failNode;
			while (1) {
				AcNode *upper_acNode = upper_fail->chdNode;
				while (upper_acNode != NULL) {
					if (upper_acNode->contId == tmp_acNode->contId) break;
					upper_acNode = upper_acNode->broNode;
				}

				if (upper_acNode != NULL) {
					tmp_acNode->failNode = upper_acNode; // tmp_acNode is the child node.
					break;
				}

				if (upper_fail->root == -1) {
					upper_fail = upper_fail->failNode; // -1 is not root.
				} else {
					tmp_acNode->failNode = upper_fail;
					break;
				}
			}
			tmp_ac = (AcQueue *)malloc(sizeof(AcQueue));
			tmp_ac->ac = tmp_acNode;
			tmp_ac->next = NULL;
			tail_ac->next = tmp_ac;
			tail_ac = tmp_ac;
			tmp_acNode = tmp_acNode->broNode;
		}
		// free the tmp(the original flag_ac) of AcQueue. Not the AcNode!
		free(tmp);
		flag_ac = flag_ac->next;
	}
}

static void transAc(RuleSetRoot *rsr)
{
	int i, j;
	for (i = 0; i < MAX_STATE; i++) {
		for (j = 0; j < 256; j++) {
			rsr->acGPU[257 * i + j] = rsr->acArray[i][j];
		}
		rsr->acGPU[257 * i + 256] = rsr->failure[i];
	}
}

static void transContPattMatch(RuleSetRoot *rsr, AcNode *acnode)
{
	rsr->contPattGPU[acnode->nodeNum].contId = acnode->contId;
	if (acnode->chdNode != NULL) rsr->contPattGPU[acnode->nodeNum].chdNode = acnode->chdNode->nodeNum;
	if (acnode->broNode != NULL) rsr->contPattGPU[acnode->nodeNum].broNode = acnode->broNode->nodeNum;
	if (rsr->contPattMatch->failNode != NULL) rsr->contPattGPU[acnode->nodeNum].failNode = acnode->failNode->nodeNum;
	rsr->contPattGPU[acnode->nodeNum].pattId = acnode->pattId;
	rsr->contPattGPU[acnode->nodeNum].root = acnode->root;
	
	if (acnode->broNode != NULL) transContPattMatch(rsr, acnode->broNode);
	if (acnode->chdNode != NULL) transContPattMatch(rsr, acnode->chdNode);
}

static void transmission(RuleSetRoot *rsr)
{
	// fit acArray in GPU
	transAc(rsr);
	
	// fit ContPattMatch in GPU
	int i;
/*	rsr->cpGPU = (uint16_t **)malloc(rsr->pattNum * sizeof(uint16_t *));
	for (i = 0; i < rsr->pattNum; i++)
	{
		rsr->cpGPU[i] = (uint16_t *)malloc(11 * sizeof(uint16_t));
		memset(rsr->cpGPU[i], 0, 11 * sizeof(uint16_t));
	}
	transCp(rsr, rsr->contPattMatch, 0);*/

	rsr->contPattGPU = (AcNodeGPU *)malloc(rsr->nodeNum * sizeof(AcNodeGPU));
	for (i = 0; i < rsr->nodeNum; i++) {
		rsr->contPattGPU[i].contId = -1;
		rsr->contPattGPU[i].chdNode = -1;
		rsr->contPattGPU[i].broNode = -1;
		rsr->contPattGPU[i].failNode = -1;
		rsr->contPattGPU[i].pattId = -1;
		rsr->contPattGPU[i].root = -1;
	}

	transContPattMatch(rsr, rsr->contPattMatch);
}

static void createarray(RuleTreeRoot *treeroot)
{
	if (treeroot->rtn == NULL) {
		return;
	} else {	
		RuleSetRoot *rsr = (RuleSetRoot *)malloc(sizeof(RuleSetRoot));
		memset(rsr->acArray, 0, MAX_STATE * 256 * sizeof(uint16_t));
		memset(rsr->failure, 0, MAX_STATE * sizeof(int16_t));
		memset(rsr->acGPU, 0, MAX_STATE * 257 * sizeof(uint16_t));
	//	rsr->cpGPU = NULL;
	//	rsr->pattNum = 0;
		rsr->contPattMatch = (AcNode *)malloc(sizeof(AcNode));
		rsr->contPattMatch->str = NULL;
		rsr->contPattMatch->contId = -1; // The root of ContPattMatch List.
		rsr->contPattMatch->chdNode = NULL;
		rsr->contPattMatch->broNode = NULL;
		rsr->contPattMatch->failNode = rsr->contPattMatch;
		rsr->contPattMatch->pattId = 0; // No pattern here.
		rsr->contPattMatch->root = 1; // true = 1, meaning root = 1.
		rsr->contPattMatch->nodeNum = 0;
		rsr->contPattGPU = NULL;
		rsr->nodeNum = 0;

		treeroot->rsr = rsr;
		newState = 0;
		nodeNum = 0;
	//	pattNum = 0;
		RuleTreeNode *treenode;
		treenode = treeroot->rtn;

		while(treenode != NULL) {
			OptTreeNode *optnode;
			optnode = treenode->down;

			while(optnode != NULL) {
				int pattId = optnode->evalIndex;
				AcNode *tmp_acNode;
				tmp_acNode = rsr->contPattMatch;
				OptFpList *fplist;
				fplist = optnode->opt_func;

				while(fplist != NULL) {
					buildACarray(rsr, fplist);
					tmp_acNode = buildContPattMatch(tmp_acNode, fplist);
					fplist = fplist->next;
				}

				if (tmp_acNode->pattId != 0) {
				//	printf("!!!!!!Error: Different patterns have the same rule options!---%d\n", tmp_acNode->pattId);
					optnode->evalIndex = tmp_acNode->pattId; // Different patterns have the same rule options!!!
				} else {
					tmp_acNode->pattId = pattId;
	//				pattNum++;
				}
				optnode = optnode->next;
			}
			treenode = treenode->right;
		}
		buildfailfunc(rsr);
		buildfailContPattMatch(rsr->contPattMatch);
		
		rsr->nodeNum = nodeNum + 1;
	//	rsr->pattNum = pattNum;
		
		transmission(rsr);
	}
}

static void create(RuleListRoot *rulelistroot)
{
	int i;
	for (i = 0; i < MAX_PORTS; i++) {
		createarray(rulelistroot->prmSrcGroup[i]);
		createarray(rulelistroot->prmDstGroup[i]);
	}
	createarray(rulelistroot->prmGeneric);
}

void precreatearray(ListRoot *listroot)
{
	create(listroot->IpListRoot);
	create(listroot->TcpListRoot);
	create(listroot->UdpListRoot);
	create(listroot->IcmpListRoot);
}

static RuleTreeNode *parseruleheader(char *ch_buffer)
{
	char *p;
	char *header;
	header = ch_buffer;

	// Check ALERT part of rule header
	p = strstr(header, ALERT);
	if (p == NULL) {
		printf("!!!!!!Error: there is no ALERT in rule header!\n");
		return NULL;
	}

	while(p != header) {
		if (*p != ' ' && *p != '\t') {
			printf("!!!!!!Error: the beginning of ch_buffer of rule header is illegal!\n");
			return NULL;
		}
		header++;
	}
	
	// Check PROTOCOL part of rule header
	// Note: not include ip
	p = strstr(header, TCP);
	int type = 0;
	if (p == NULL) {
		p = strstr(header, UDP);
		type = 1;
	}

	if (p == NULL) {
		p = strstr(header, ICMP);
		type = 2;
	}

	header += 5;
	if (p == header + 1 && *header == ' ') {
		header += 1; // space after ALERT
	} else {
		printf("!!!!!!Error: the character after ALERT is illegal!\n");
		return NULL;
	}

	if (type == 2) {
		header += 4;
	} else {
		header += 3;
	}

	if (*header == ' ') {
		header += 1; // space after PROTOCOL
	} else {
		printf("!!!!!!Error: the character after PROTOCOL is illegal!\n");
		return NULL;
	}

	int i;
	int hport[MAX_PORTS];
	int lport[MAX_PORTS];
	int direction = -1;
	int flag_hport = 0;
	int flag_lport = 0;
	memset(hport, -1, MAX_PORTS * sizeof(int));
	memset(lport, -1, MAX_PORTS * sizeof(int));

	for (i = 0; i < 2; i++) {
		// Check IP part of rule header
		p = strstr(header, " "); // space after IP
		header = p + 1; // Ignore src ip

		// Check PORT part of rule header
		p = header;
		if (*p == 'a' && *(p + 1) == 'n' && *(p + 2) == 'y') {
			lport[flag_lport++] = 0;
			p += 3;
		} else if (*p == '[') {
			p++;
			int tmp_hport = 0;
			int flag = 0;

			while(*p != ']') {
				if (isdigit((int) *p)) {
					tmp_hport = 10 * tmp_hport + (*p - '0');
					p++;
					while(isdigit((int) *p)) {
						tmp_hport = 10 * tmp_hport + (*p - '0');
						p++;
					}
				} else {
					// ignore the form like [variable, num]
					p = strstr(p, ",");
				}

				if (*p == ',') {
					flag = -1;				
					p++;
				} else if (*p == ':') {
					flag = -2;
					if (*(p+1) == ',') {
						flag = -3;
						p++;
					}
					p++;
				} else {
					flag = -1;
				}

				switch(flag) {
					case -1:
						lport[flag_lport++] = tmp_hport;
						break;
					case -2:
						lport[flag_lport++] = tmp_hport;
						lport[flag_lport++] = flag; 
						break;
					case -3:
						lport[flag_lport++] = tmp_hport;
						lport[flag_lport++] = flag;
						break;
					default:
						break;
				}
				tmp_hport = 0;
			}
			p++;
		} else if (isdigit((int) *p)) {
			int tmp_hport = 0;
			tmp_hport = 10 * tmp_hport + (*p - '0');
			p++;

			while(isdigit((int) *p)) {
				tmp_hport = 10 * tmp_hport + (*p - '0');
				p++;
			}
			lport[flag_lport++] = tmp_hport;
			if (*p == ':') {
				p++;
				lport[flag_lport++] = -2;
				tmp_hport = 0;
				if (isdigit((int) *p)) {
					tmp_hport = 10 * tmp_hport + (*p - '0');
					p++;
					while(isdigit((int) *p)) {
						tmp_hport = 10 * tmp_hport + (*p - '0');
						p++;
					}
					lport[flag_lport++] = tmp_hport;
				}
			}
		} else { // PORT is a varialable.
			lport[flag_lport++] = 0;
			p = strstr(header, " ");
		}

		if (i == 0) {
			memcpy(hport, lport, flag_lport * sizeof(int));
			flag_hport = flag_lport;
			// reset lport & flag_lport
			memset(lport, -1, flag_lport * sizeof(int));
			flag_lport = 0;
		}

		if (*p != ' ') { // space after the PORT number
			if (i == 0) printf("!!!!!!Error: the character after first PORT number is illegal!\n");
			if (i == 1) printf("!!!!!!Error: the character after second PORT number is illegal!\n");
			return NULL;
		}

		if (i == 0) header = p + 1; 
		p = header;

		// Check the direction only when i == 0
		if (i == 0) {
			if (*p == '-' && *(p + 1) == '>') {
				direction = 1;
			} else if (*p == '<' && *(p + 1) == '-') {
				direction = 2;
			} else if (*p == '<' && *(p + 1) == '>') {
				direction = 4;
			} else {
				printf("!!!!!!Error: the character of DIRECTION is illegal!\n");
				return NULL;
			}

			if (*(p + 2) != ' ') { // space after the DIRECTION
				printf("!!!!!!Error: the character after DIRECTION is illegal!\n");
				return NULL;
			} else {
				header = p + 3;
			}
		}
	}
	/*
	printf("############################################\n");
	printf("alert type: %d\n", type);
	printf("direction type: %d\n", direction);
	printf("first port number\t");
	for (i = 0; i < flag_hport; i++) printf("%d\t", hport[i]);
	printf("\nsecond port number\t");
	for (i = 0; i < flag_lport; i++) printf("%d\t", lport[i]);
	printf("\n");
	*/

	//configure array hport and array lport as unsigned_hport and unsigned_lport
	uint16_t unsigned_hport[MAX_PORTS];
	uint16_t unsigned_lport[MAX_PORTS];
	memset(unsigned_hport, 0, MAX_PORTS * sizeof(uint16_t));
	memset(unsigned_lport, 0, MAX_PORTS * sizeof(uint16_t));

	int j = 0;
	for (i = 0; i < flag_hport; i++) {
		if (hport[i] == 0) {
			;
		} else if (hport[i] == -2 && hport[i + 1] != -1) {
			while(unsigned_hport[j - 1] != hport[i + 1] - 1) {
				unsigned_hport[j] = unsigned_hport[j - 1] + 1;	
				j++;
			}
		} else if (hport[i] == -3 || (hport[i] == -2 && hport[i + 1] == -1)) {
			while(unsigned_hport[j - 1] != (MAX_PORTS - 1)) {
				unsigned_hport[j] = unsigned_hport[j - 1] + 1;
				j++;
			}
		} else {
			unsigned_hport[j++] = hport[i];
		}
	}

	/*
	printf("first port number\t");
	if (j == 0) printf("%u\t", unsigned_hport[j]);
	for (i = 0; i < j; i++) printf("%u\t", unsigned_hport[i]); 
	*/

	j = 0;
	for (i = 0; i < flag_lport; i++) {
		if (lport[i] == 0) {
			;
		} else if (lport[i] == -2 && lport[i + 1] != -1) {
			while(unsigned_lport[j - 1] != lport[i + 1] - 1) {	
				unsigned_lport[j] = unsigned_lport[j - 1] + 1; 
				j++;
			}
		} else if (lport[i] == -3 || (lport[i] == -2 && lport[i + 1] == -1)) {
			while(unsigned_lport[j - 1] != (MAX_PORTS - 1)) {
				unsigned_lport[j] = unsigned_lport[j - 1] + 1;
				j++;
			}
		} else {
			unsigned_lport[j++] = lport[i];
		}
	}

	/*
	printf("\nsecond port number\t");
	if (j == 0) printf("%u\t", unsigned_lport[j]);
	for (i = 0; i < j; i++) printf("%u\t", unsigned_lport[i]);
	printf("\n");
	*/

	RuleTreeNode *rtn = (RuleTreeNode *)malloc(sizeof(RuleTreeNode));
	rtn->type = type;
	memcpy(rtn->hdp, unsigned_hport, MAX_PORTS * sizeof(uint16_t));
	memcpy(rtn->ldp, unsigned_lport, MAX_PORTS * sizeof(uint16_t));
	rtn->flags = direction;
	rtn->right = NULL;
	rtn->down = NULL;

	return rtn;
}

static RuleTreeNode *parseruleoption(RuleTreeNode *rtn, char *ch_buffer, int id)
{
	//printf("------Start of rule option of %d\n", id);

	OptTreeNode *opt_node;
	opt_node = (OptTreeNode *)malloc(sizeof(OptTreeNode));
	opt_node->opt_func = NULL;
	opt_node->type = rtn->type;
	opt_node->evalIndex = id;
	opt_node->msg = NULL;
	opt_node->next = NULL;
	opt_node->rtn = rtn;
	rtn->down = opt_node;

	const char *sym[SYM_NUM];
	int fsym[SYM_NUM];
	sym[0] = "msg";
	sym[1] = "content";
	sym[2] = "depth";
	sym[3] = "offset";
	sym[4] = "within";
	sym[5] = "distance";
	sym[6] = "nocase";
	fsym[0] = MSG;
	fsym[1] = CONTENT;
	fsym[2] = DEPTH;
	fsym[3] = OFFSET;
	fsym[4] = WITHIN;
	fsym[5] = DISTANCE;
	fsym[6] = NOCASE;

	char *p1;
	char *p2;
	char *option;
	option = ch_buffer;
	while(*option != '\0') {
		int len;
		char *pChar;
		p1 = strstr(option, ":");
		p2 = strstr(option, ";");
		if (p1 < p2) {
			len = p1 - option;
			pChar = (char *)malloc((len + 1) * sizeof(char));
		} else {
			len = p2 - option;
			pChar = (char *)malloc((len + 1) * sizeof(char));
		}
		memcpy(pChar, option, len * sizeof(char));
		pChar[len] = '\0';

		int i;
		for (i = 0; i < SYM_NUM; i++) {
			if (!strcmp(pChar, sym[i])) break; // pattern matching succeed!
		}
		// free(pChar); (1)
		
		if (i == SYM_NUM) {
			option = p2 + 1; // jump to the character after ';'
			// printf("******Ignore part of rule option: %s\n", pChar);
			free(pChar); // replace free (1)
		} else {
			free(pChar); // replace free (1)
			
			if (i == 0 || i == 1) { // "msg" || "content"
				option = p1 + 2; // suppose the form is name:value without any space
				int exclamation = 0;
				if (*(option - 1) == '!') { // for content:!"..."
					option++; 
					exclamation = 1;
				}
				len = p2 - option - 1;
				pChar = (char *)malloc((len + 1) * sizeof(char));
				char *ptr_vertline = NULL;
				ptr_vertline = strstr(option, "|");

				if (ptr_vertline == NULL || (ptr_vertline != NULL && (ptr_vertline - option) > len)) {
					memcpy(pChar, option, len * sizeof(char));
					pChar[len] = '\0';
				} else {
					int j = 0, k = 0;
					int flag_vertline = 0;
					while (k < len) // ignore bytecode
					{
						if (option[k] == '|') flag_vertline++;
						else if (option[k] != '|' && flag_vertline % 2 == 0) pChar[j++] = option[k];
						//else ; 
						k++;
					}
					if (j == 0) pChar[j++] = 't'; // ignore bytecode
					pChar[j] = '\0';
					len = j;
				}

				if (i == 0)
					opt_node->msg = pChar;

				if (i == 1) {
					OptFpList *opt_fp;
					opt_fp = (OptFpList *)malloc(sizeof(OptFpList));
					opt_fp->context = pChar;
					opt_fp->index = contIndex;
					opt_fp->depth = 0;
					opt_fp->offset = 0;
					opt_fp->distance = 0;
					opt_fp->within = 0;
					opt_fp->flags = 0;
					opt_fp->next = NULL;

					OptFpList *tmp_ptr_opt_fp;
					tmp_ptr_opt_fp = opt_node->opt_func;
					opt_node->opt_func = opt_fp;
					opt_fp->next = tmp_ptr_opt_fp;

					if (exclamation == 1) opt_node->opt_func->flags += fsym[i];

					contIndex++;
				}

				/*
				if (i == 0) printf("msg: %s\n", pChar);
				if (i == 1 && exclamation == 0) printf("content: %s\n", pChar);
				if (i == 1 && exclamation == 1) printf("content:!\" %s\"\n", pChar);
				*/
			} else if (i == 6) { // nocase
				opt_node->opt_func->flags += fsym[i];
				// printf("nocase\n");
			} else { // others
				// change the flags
				opt_node->opt_func->flags += fsym[i];

				// attain the value
				option = p1 + 1; // suppose the for is name:value without any space
				len = p2 - option;
				pChar = (char *)malloc((len + 1) * sizeof(char));
				memcpy(pChar, option, len * sizeof(char));
				pChar[len] = '\0';
				int j;
				int tmp = 0;

				if (pChar[0] == '!') j = 1; // for the case like name:!value
				else j = 0;

				for (; pChar[j] != '\0'; j++) {
					tmp = 10 * tmp + (pChar[j] - '0');
				}

				if (pChar[0] == '!') tmp = -tmp;

				switch(i) {
					case 2: // depth
						opt_node->opt_func->depth = tmp;
						break;
					case 3: // offset
						opt_node->opt_func->offset = tmp;
						break;
					case 4: // within
						opt_node->opt_func->within = tmp;
						break;
					case 5: // distance
						opt_node->opt_func->distance = tmp;
						break;
					default:
						printf("!!!!!!Error: i of opt_func is illegal -- %d\n", i);
						return NULL;
				}

				/*
				switch(i) {
					case 2: // depth
						printf("depth: %d\n", tmp);
						break;
					case 3: // offset
						printf("offset: %d\n", tmp);
						break;
					case 4: // within
						printf("within: %d\n", tmp);
						break;
					case 5: // distance
						printf("distance: %d\n", tmp);
						break;
					default:
						printf("!!!!!!Error: i of opt_func is illegal -- %d\n", i);
						return NULL;
				}
				*/
			}
			option = p2 + 1;
		}
		if (*option == ' ') option++;
	}
	return rtn;
}

static RuleTreeNode *duplicatertn(RuleTreeNode *rtn)
{
	int i;
	RuleTreeNode *ptr_rtn;
	ptr_rtn = (RuleTreeNode *)malloc(sizeof(RuleTreeNode));
	ptr_rtn->type = rtn->type;
	// ignore sip & dip ptr_rtn->sip = rtn->sip;
	for (i = 0; i < MAX_PORTS; i++) {
		ptr_rtn->hdp[i] = rtn->hdp[i];
		ptr_rtn->ldp[i] = rtn->ldp[i];
	}
	ptr_rtn->flags = rtn->flags;
	ptr_rtn->right = NULL;
	ptr_rtn->down = NULL;
	
	OptTreeNode *opt;
	OptTreeNode *ptr_opt;
	OptTreeNode *tmp_opt;

	tmp_opt = rtn->down;

	while(tmp_opt != NULL) {
		ptr_opt = (OptTreeNode *)malloc(sizeof(OptTreeNode));
		ptr_opt->opt_func = NULL;
		ptr_opt->type = tmp_opt->type;
		ptr_opt->evalIndex = tmp_opt->evalIndex;
		ptr_opt->msg = tmp_opt->msg;
		ptr_opt->next = NULL;
		ptr_opt->rtn = ptr_rtn;

		OptFpList *fp;
		OptFpList *ptr_fp;
		OptFpList *tmp_fp;

		tmp_fp = tmp_opt->opt_func;

		while(tmp_fp != NULL) {
			ptr_fp = (OptFpList *)malloc(sizeof(OptFpList));
			ptr_fp->context = tmp_fp->context;
			ptr_fp->index = tmp_fp->index;
			ptr_fp->depth = tmp_fp->depth;
			ptr_fp->offset = tmp_fp->offset;
			ptr_fp->distance = tmp_fp->distance;
			ptr_fp->within = tmp_fp->within;
			ptr_fp->flags = tmp_fp->flags;
			ptr_fp->next = NULL;

			if (ptr_opt->opt_func == NULL) ptr_opt->opt_func = ptr_fp;
			else fp->next = ptr_fp;
			fp = ptr_fp;

			tmp_fp = tmp_fp->next;
		}

		if (ptr_rtn->down == NULL) ptr_rtn->down = ptr_opt;
		else opt->next = ptr_opt;
		opt = ptr_opt;

		tmp_opt = tmp_opt->next;
	}
	return ptr_rtn;
}

static void assemblelistroot(ListRoot *listroot, RuleTreeNode *rtn)
{
	// in terms of type of PROTOCOL, put parsed rule in the listroot
	RuleListRoot *rulelistroot;
	switch(rtn->type) {
		case 0:
			rulelistroot = listroot->TcpListRoot;
			break;
		case 1:
			rulelistroot = listroot->UdpListRoot;
			break;
		case 2:
			rulelistroot = listroot->IcmpListRoot;
			break;
		default:
			printf("!!!!!!Error: alert type is not Tcp or Udp or Icmp!\n");
			break;
	}

	RuleTreeRoot *pRoot;
	RuleTreeNode *ptr_rtn;
	int flag = 0;

	if (rtn->flags & 0x00000007 & 0x01 || rtn->flags & 0x00000007 & 0x04) { // direction is "->" || "<>"
		if (rtn->hdp[0] == 0 && rtn->ldp[0] != 0 && rtn->ldp[1] == 0) {
			pRoot = rulelistroot->prmDstGroup[rtn->ldp[0]];
		} else if (rtn->hdp[0] != 0 && rtn->hdp[1] == 0 && rtn->ldp[0] == 0) {
			pRoot = rulelistroot->prmSrcGroup[rtn->hdp[0]];
		} else if (rtn->hdp[0] != 0 && rtn->hdp[1] == 0 && rtn->ldp[0] != 0 && rtn->ldp[1] == 0) {
			pRoot = rulelistroot->prmDstGroup[rtn->ldp[0]];
			flag = 1;
		} else {
			pRoot = rulelistroot->prmGeneric;
		}
		
		ptr_rtn = pRoot->rtn;
		pRoot->rtn = rtn;
		rtn->right = ptr_rtn;

		if (flag == 1) {
			pRoot = rulelistroot->prmSrcGroup[rtn->hdp[0]];

			rtn = duplicatertn(rtn);

			ptr_rtn = pRoot->rtn;
			pRoot->rtn = rtn;
			rtn->right = ptr_rtn;
		}
	}

	if (rtn->flags & 0x00000007 & 0x02 || rtn->flags & 0x00000007 & 0x04) { // direction is "<-" || "<>"
		if (rtn->flags & 0x00000007 & 0x04) {
			rtn = duplicatertn(rtn);
		}

		if (rtn->hdp[0] == 0 && rtn->ldp[0] != 0 && rtn->ldp[1] == 0) {
			pRoot = rulelistroot->prmSrcGroup[rtn->ldp[0]];
		} else if (rtn->hdp[0] != 0 && rtn->hdp[1] == 0 && rtn->ldp[0] == 0) {
			pRoot = rulelistroot->prmDstGroup[rtn->hdp[0]];
		} else if (rtn->hdp[0] != 0 && rtn->hdp[1] == 0 && rtn->ldp[0] != 0 && rtn->ldp[1] == 0) {
			pRoot = rulelistroot->prmDstGroup[rtn->ldp[0]];
			flag = 1;
		} else {
			pRoot = rulelistroot->prmGeneric;
		}
		
		ptr_rtn = pRoot->rtn;
		pRoot->rtn = rtn;
		rtn->right = ptr_rtn;

		if (flag == 1) {
			pRoot = rulelistroot->prmSrcGroup[rtn->hdp[0]];

			rtn = duplicatertn(rtn);

			ptr_rtn = pRoot->rtn;
			pRoot->rtn = rtn;
			rtn->right = ptr_rtn;
		}
	}
}

static int readfile(ListRoot* listroot, const char *pFName)
{
	FILE *pFile = fopen(pFName, "r");
	if (pFile == NULL) {
		printf("Cannot open file %s\n", pFName);
		exit(0);
	}
	
	int lNum = 0;
	int patternNum = 0;
	char ch;
	char ch_buffer[MAX_CH_BUF];
	int ch_buf_num = 0;
	int hd_op_flag = 0;

	fscanf(pFile, "%c", &ch);

	while(!feof(pFile)) {
		lNum++;

		if (ch == '#') {
			while(ch != '\n') {
				fscanf(pFile, "%c", &ch);
			}
			// printf("------Ignore Line %d\n", lNum);
		} else {
			patternNum++;
			RuleTreeNode *return_rtn;
			hd_op_flag = 0;
			ch_buf_num = 0;
			memset(ch_buffer, 0, MAX_CH_BUF * sizeof(char));
			while(1) {
			    if (ch_buf_num > MAX_CH_BUF) {
				    printf("!!!!!!Error: ch_buffer has overflowed!\n");
					return 0;
				}

				if (hd_op_flag == 0) { // for rule header part{
					switch(ch) {
						case '\n':
							// printf("------Ignore Line %d\n", lNum);
							hd_op_flag = -1;
							break;
						case '(':
							hd_op_flag = 1;
							break;
						default:
							ch_buffer[ch_buf_num++] = ch;
							break;
					}
				} else if (hd_op_flag == 2) { // for rule option part
					switch(ch) {
						case '\n':
							hd_op_flag = 3;
							break;
						case '(': // suppose there is no '(' | or ')' in content 
							ch_buffer[ch_buf_num++] = ch;
							break;
						case ')':
							ch_buffer[ch_buf_num++] = ch;
							break;
						default:
							ch_buffer[ch_buf_num++] = ch;
							break;
					}
				}

				if (hd_op_flag == 1) {
					// parse ch_buffer for rule header part
					return_rtn = parseruleheader(ch_buffer); 
					// printf("%s\n", ch_buffer);
					
					hd_op_flag = 2;
					ch_buf_num = 0;
					memset(ch_buffer, 0, MAX_CH_BUF * sizeof(char));
				} else if (hd_op_flag == 3) {
					// parse ch_buffer for rule option part
					ch_buffer[ch_buf_num - 1] = '\0'; // eliminate the final ')'
					return_rtn = parseruleoption(return_rtn, ch_buffer, patternNum);
					assemblelistroot(listroot, return_rtn);
					// printf("%s\n", ch_buffer);
					hd_op_flag = -1;
				}
				
				if (hd_op_flag == -1) break;

				fscanf(pFile, "%c", &ch);
			}
			// printf("******Finish Line %d for pattern %d\n", lNum, patternNum);
		}
		fscanf(pFile, "%c", &ch);
	}
	fclose(pFile);
	return 1;
}

ListRoot *configrules(const char *filename)
{
	ListRoot *listroot = (ListRoot *)malloc(sizeof(ListRoot));
	listroot->IpListRoot = (RuleListRoot *)malloc(sizeof(RuleListRoot));
	listroot->TcpListRoot = (RuleListRoot *)malloc(sizeof(RuleListRoot));
	listroot->UdpListRoot = (RuleListRoot *)malloc(sizeof(RuleListRoot));
	listroot->IcmpListRoot = (RuleListRoot *)malloc(sizeof(RuleListRoot));

	int i;
	RuleListRoot *rulelistroot;
	rulelistroot = listroot->IpListRoot;
	for (i = 0; i < MAX_PORTS; i++) {
		rulelistroot->prmDstGroup[i] = (RuleTreeRoot *)malloc(sizeof(RuleTreeRoot));
		rulelistroot->prmDstGroup[i]->rtn = NULL;
		rulelistroot->prmDstGroup[i]->rsr = NULL;

		rulelistroot->prmSrcGroup[i] = (RuleTreeRoot *)malloc(sizeof(RuleTreeRoot));
		rulelistroot->prmSrcGroup[i]->rtn = NULL;
		rulelistroot->prmSrcGroup[i]->rsr = NULL;
	}

	rulelistroot = listroot->TcpListRoot;
	for (i = 0; i < MAX_PORTS; i++) {
		rulelistroot->prmDstGroup[i] = (RuleTreeRoot *)malloc(sizeof(RuleTreeRoot));
		rulelistroot->prmDstGroup[i]->rtn = NULL;
		rulelistroot->prmDstGroup[i]->rsr = NULL;

		rulelistroot->prmSrcGroup[i] = (RuleTreeRoot *)malloc(sizeof(RuleTreeRoot));
		rulelistroot->prmSrcGroup[i]->rtn = NULL;
		rulelistroot->prmSrcGroup[i]->rsr = NULL;
	}

	rulelistroot = listroot->UdpListRoot;
	for (i = 0; i < MAX_PORTS; i++) {
		rulelistroot->prmDstGroup[i] = (RuleTreeRoot *)malloc(sizeof(RuleTreeRoot));
		rulelistroot->prmDstGroup[i]->rtn = NULL;
		rulelistroot->prmDstGroup[i]->rsr = NULL;

		rulelistroot->prmSrcGroup[i] = (RuleTreeRoot *)malloc(sizeof(RuleTreeRoot));
		rulelistroot->prmSrcGroup[i]->rtn = NULL;
		rulelistroot->prmSrcGroup[i]->rsr = NULL;
	}

	rulelistroot = listroot->IcmpListRoot;
	for (i = 0; i < MAX_PORTS; i++) {
		rulelistroot->prmDstGroup[i] = (RuleTreeRoot *)malloc(sizeof(RuleTreeRoot));
		rulelistroot->prmDstGroup[i]->rtn = NULL;
		rulelistroot->prmDstGroup[i]->rsr = NULL;

		rulelistroot->prmSrcGroup[i] = (RuleTreeRoot *)malloc(sizeof(RuleTreeRoot));
		rulelistroot->prmSrcGroup[i]->rtn = NULL;
		rulelistroot->prmSrcGroup[i]->rsr = NULL;
	}

	listroot->IpListRoot->prmGeneric = (RuleTreeRoot *)malloc(sizeof(RuleTreeRoot));
	listroot->TcpListRoot->prmGeneric = (RuleTreeRoot *)malloc(sizeof(RuleTreeRoot));
	listroot->UdpListRoot->prmGeneric = (RuleTreeRoot *)malloc(sizeof(RuleTreeRoot));
	listroot->IcmpListRoot->prmGeneric = (RuleTreeRoot *)malloc(sizeof(RuleTreeRoot));

	RuleTreeRoot *ruletreeroot = listroot->IpListRoot->prmGeneric;
	ruletreeroot->rtn = NULL;
	ruletreeroot->rsr = NULL;

	ruletreeroot = listroot->TcpListRoot->prmGeneric;
	ruletreeroot->rtn = NULL;
	ruletreeroot->rsr = NULL;

	ruletreeroot = listroot->UdpListRoot->prmGeneric;
	ruletreeroot->rtn = NULL;
	ruletreeroot->rsr = NULL;

	ruletreeroot = listroot->IcmpListRoot->prmGeneric;
	ruletreeroot->rtn = NULL;
	ruletreeroot->rsr = NULL;

	// read rule file and configure rules
	if (!readfile(listroot, filename)) {
		printf("\n\n$$ Error(s)!");
		return NULL;
	}

/*	RuleTreeNode *ptr_node;
	RuleTreeNode *free_ptr_node;
	ptr_node = ruletreeroot->rtn;
	free_ptr_node = ptr_node;
	while(1)
	{
		free_ptr_node = ptr_node;
		if (ptr_node == NULL)
		{
			printf("End! (ptr_node == NULL)\n");
			return 0;
		}
		printf("alert type %d\n", ptr_node->type);
		printf("direction %d\n", ptr_node->flags);
		printf("first port number\t");
		int i = 0;
		while(1)
		{
			if (ptr_node->hdp[i] == -1) break;
			printf("%d\t", ptr_node->hdp[i]);
			i++;
		}
		printf("\nsecond port number\t");
		i = 0;
		while(1)
		{
			if (ptr_node->ldp[i] == -1) break;
			printf("%d\t", ptr_node->ldp[i]);
			i++;
		}
		printf("\n");
		ptr_node = ptr_node->right;
		free(free_ptr_node);
	}*/

	return listroot;
}
