#include "ArvoreB.h"
#include "NoB.h"
#include <iostream>

using namespace std;

ArvoreB::ArvoreB(int dMin)
{
    raiz = NULL;
    this->ordem=2*dMin;
}

ArvoreB::ArvoreB(int dMin, HashTable *t)
{
    raiz = NULL;
    this->ordem=2*dMin;
    this->t=t;

}

int ArvoreB::busca(int val)
{
    if(raiz==NULL)
        return 0;
    else{
        NoB* aux= new NoB(ordem,t);
        aux=raiz;
        return aux->busca(val);
    }
       // return raiz->busca(val);
}

void ArvoreB::insere(int val){

    if(raiz == NULL){
        raiz = new NoB(ordem);
        raiz->setFolha(true);
        raiz->setChave(val,0);
        raiz->setN(1);

    }
    else{
        if(raiz->getN()==ordem-1){   //raiz esta cheia
            NoB* novo= new NoB(ordem);
            novo->setFolha(false);
            novo->setFilho(raiz,0);
            novo->divideFilho(novo,0);
            int i = 0;
            if(novo->getChave(0) < val )
                i++;
            novo->getFilho(i)->insereComEspaco(val);
            raiz=novo;
    }
        else{
            raiz->insereComEspaco(val);
        }

    }
}

void ArvoreB::imprime(ostream & o){
    if(raiz!=NULL)
        raiz->imprime(o);
}

