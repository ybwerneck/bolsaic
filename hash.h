#ifndef HASH_H_INCLUDED
#define HASH_H_INCLUDED
#include <string>
#include <iostream>
#include <fstream>
#include "NodeHT.h"


typedef std::string string;
class HashTable
{
public:
    int getTam() { return tam; }
    int getIndice(string, int);
    HashTable(int tam);
    void insere(int codcidade, string data, string cidade, string estado, int casos, int mortes) {
        NodeHT *n= new NodeHT(codcidade, data, cidade, estado, casos, mortes);
        insereaux(n);
    };
    void insere(NodeHT* n) {
        insere(n->codcidade,n-> data,n-> cidade,n-> estado,n-> casos,n-> mortes);
    };
    void print(std::ostream&);
    NodeHT* operator[](int i){
    if(tabela[i]!=NULL){
        NodeHT*a = tabela[i];
        return a;
    }
    return NULL;
    }
   NodeHT* busca(string data ,int codcidade);

private:
    int tam;
    NodeHT **tabela;
    void insereaux(NodeHT*);
    int funct(string,int,int);


};



#endif // HASH_H_INCLUDED
