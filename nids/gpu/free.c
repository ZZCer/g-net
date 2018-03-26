#include <stdio.h>
#include <stdlib.h>

#include "rules.h"

void freeoptfplist(OptFpList *fp)
{
	OptFpList *ptr_fp;

	while (fp != NULL) {
		ptr_fp = fp;
		fp = ptr_fp->next;
		free(ptr_fp);
	}
}

void freeopttreenode(OptTreeNode *opt)
{
	OptTreeNode *ptr_opt;

	while (opt != NULL) {
		ptr_opt = opt;
		opt = ptr_opt->next;

		if (ptr_opt->opt_func != NULL) freeoptfplist(ptr_opt->opt_func);

		free(ptr_opt);
	}
}

void freeacnode(AcNode *contPattMatch)
{
	if (contPattMatch != NULL) {
		AcNode *tmp, *chd;
		tmp = contPattMatch->broNode;
		chd = contPattMatch->chdNode;
		free(contPattMatch);
		if (tmp != NULL) freeacnode(tmp);
		if (chd != NULL) freeacnode(chd);
	}
}

void freerulesetroot(RuleSetRoot *rsr)
{
	freeacnode(rsr->contPattMatch);
	free(rsr->contPattGPU);
	free(rsr);
}

void freeruletreenode(RuleTreeNode *rtn)
{
	RuleTreeNode *ptr_rtn;
	while (rtn != NULL) {
		ptr_rtn = rtn;
		rtn = ptr_rtn->right;

		if (ptr_rtn->down != NULL) freeopttreenode(ptr_rtn->down);

		free(ptr_rtn);
	}
}

void freerulelistroot(RuleListRoot *rulelistroot)
{
	RuleTreeRoot *ptr_treeroot;

	int i;
	for (i = 0; i < MAX_PORTS; i++) {
		ptr_treeroot = rulelistroot->prmSrcGroup[i];
		if (ptr_treeroot->rtn != NULL) freeruletreenode(ptr_treeroot->rtn);
		if (ptr_treeroot->rsr != NULL) freerulesetroot(ptr_treeroot->rsr);
		free(ptr_treeroot);

		ptr_treeroot = rulelistroot->prmDstGroup[i];
		if (ptr_treeroot->rtn != NULL) freeruletreenode(ptr_treeroot->rtn);
		if (ptr_treeroot->rsr != NULL) freerulesetroot(ptr_treeroot->rsr);
		free(ptr_treeroot);
	}

	ptr_treeroot = rulelistroot->prmGeneric;
	if (ptr_treeroot->rtn != NULL) freeruletreenode(ptr_treeroot->rtn);
	if (ptr_treeroot->rsr != NULL) freerulesetroot(ptr_treeroot->rsr);

	free(ptr_treeroot);
}

void freeall(ListRoot *listroot)
{
	RuleListRoot *ptr_rulelistroot;

	ptr_rulelistroot = listroot->IpListRoot;
	if (ptr_rulelistroot != NULL) freerulelistroot(ptr_rulelistroot);
	free(ptr_rulelistroot);

	ptr_rulelistroot = listroot->TcpListRoot;
	if (ptr_rulelistroot != NULL) freerulelistroot(ptr_rulelistroot);
	free(ptr_rulelistroot);

	ptr_rulelistroot = listroot->UdpListRoot;
	if (ptr_rulelistroot != NULL) freerulelistroot(ptr_rulelistroot);
	free(ptr_rulelistroot);

	ptr_rulelistroot = listroot->IcmpListRoot;
	if (ptr_rulelistroot != NULL) freerulelistroot(ptr_rulelistroot);
	free(ptr_rulelistroot);

	free(listroot);
}
