#include <stdio.h>
#include "rules.h"

void printAcNodeTree(AcNode *acNode, int i)
{
	while(acNode != NULL)
	{
//		printf("depth: %4d, contId: %4d; pattId: %4d; failId: %4d; root: %4d; str: %s\n", i, acNode->contId, acNode->pattId, acNode->failNode->contId, acNode->root, acNode->str);
//		printf("depth: %4d, contId: %4d; pattId: %4d; root: %4d; str: %s\n", i, acNode->contId, acNode->pattId, acNode->root, acNode->str);
		printf("depth: %4d", i); 
		printf(", contId: %4d", acNode->contId); 
		printf(", pattId: %4d", acNode->pattId); 
		printf(", root: %4d", acNode->root); 
		printf(", str: %s", acNode->str); 
		printf(", fail : %4d", acNode->failNode->root); 
		printf(", failId: %4d\n", acNode->failNode->contId);
		printAcNodeTree(acNode->chdNode, i + 1);
		acNode = acNode->broNode;
	}
}

void print(ListRoot *listroot)
{
	RuleSetRoot *rsr;
	rsr = listroot->TcpListRoot->prmGeneric->rsr;
	
	printf("Printf acArray...\n");
	int i, j;
	for(i = 0; i < MAX_STATE; i++)
	{
		for(j = 0; j < 256; j++)
		{
			if(rsr->acArray[i][j] != 0) printf("(%d, %d)---%d\n", i, j, rsr->acArray[i][j]);
		}
	}
	printf("Printf failure...\n");
	for(i = 0; i < MAX_STATE; i++)
	{
		if(rsr->failure[i] != 0) printf("--%d  %d\n", i, rsr->failure[i]);
	}
	printf("Printf contPattMatch...\n");
	printAcNodeTree(rsr->contPattMatch, 0);
}

void test(ListRoot *listroot)
{
	printf("\n\nTest...\n");
	print(listroot);
	printf("Test End.\n");
}
