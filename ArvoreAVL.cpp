#include <iostream>
#include "ArvoreAVL.h"

using namespace std;

ArvoreAVL::ArvoreAVL()
{
    raiz = NULL;
}

void ArvoreAVL::insere(int val)
{
    raiz = auxInsere(raiz, val);
}

NoArv* ArvoreAVL::auxInsere(NoArv *p, int val)
{
    if(p == NULL)
    {
        p = new NoArv();
        p->setInfo(val);
        p->setEsq(NULL);
        p->setDir(NULL);
        p->setFator(0);
    }
    else if(val < p->getInfo())
        p->setEsq(auxInsere(p->getEsq(), val));
    else
        p->setDir(auxInsere(p->getDir(), val));

    p->setFator(calculaAltura(p->getDir()) - calculaAltura(p->getEsq()));
    return verificaBalanceamento(p);
}

NoArv* ArvoreAVL::rotSimplesEsq(NoArv* p)
{
    NoArv *q = p->getDir();
    p->setDir(q->getEsq());
    q->setEsq(p);
    p->setFator(calculaAltura(p->getDir()) - calculaAltura(p->getEsq()) );
    q->setFator(calculaAltura(q->getDir()) - calculaAltura(q->getEsq()) );
    return q;
}

NoArv* ArvoreAVL::rotSimplesDir(NoArv* p)
{
    NoArv *q = p->getEsq();
    p->setEsq(q->getDir());
    q->setDir(p);
    p->setFator(calculaAltura(p->getDir()) - calculaAltura(p->getEsq()) );
    q->setFator(calculaAltura(q->getDir()) - calculaAltura(q->getEsq()) );
    return q;
}

NoArv* ArvoreAVL::verificaBalanceamento(NoArv *p)
{
    if(p->getFator() >= 2){
        if(p->getDir()->getFator() == 1 || p->getDir()->getFator() == 0){
            return rotSimplesEsq(p);
        }
        else if(p->getDir()->getFator() == -1){
            // rotacao dupla a esquerda
            p->setDir(rotSimplesDir(p->getDir()));
            return rotSimplesEsq(p);
        }
    }
    else if(p->getFator() ==-2){
        if(p->getEsq()->getFator() == -1 || p->getEsq()->getFator() == 0 ){
            // simples a direita
            return rotSimplesDir(p);
        }
        else if(p->getEsq()->getFator() == 1){
            //rotacao dupla a direita
             p->setEsq(rotSimplesEsq(p->getEsq()));
            return rotSimplesDir(p);
        }
    }
    return p;
}

int ArvoreAVL::calculaAltura(NoArv *p)  // calcula altura do no
{
    int i, j;
    if(p==NULL)
        return -1;

    i= calculaAltura(p->getDir());
    j= calculaAltura(p->getEsq());
    return i>j ? i+1 : j+1;
}


bool ArvoreAVL::busca(int val)
{
    return auxBusca(raiz, val);
}

bool ArvoreAVL::auxBusca(NoArv *p, int val)
{
    if(p == NULL)
        return false;
    else if(p->getInfo() == val)
        return true;
    else if(val < p->getInfo())
        return auxBusca(p->getEsq(), val);
    else
        return auxBusca(p->getDir(), val);
}


void ArvoreAVL::imprime(std::ostream& o)
{
    imprimePorNivel(raiz, 0,o);
}

void ArvoreAVL::imprimePorNivel(NoArv *p, int nivel,std::ostream &o)
{
    if(p != NULL)
    {
        o << "(" << nivel << ")";
        for(int i = 1; i <= nivel; i++)
            o << "--";
        o << p->getInfo() << endl;
        imprimePorNivel(p->getEsq(), nivel+1,o);
        imprimePorNivel(p->getDir(), nivel+1,o);
    }
}

