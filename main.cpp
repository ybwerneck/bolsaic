#include <iostream>
#include <ostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <string.h>
#include "ArvBinBusca.h"
#include "hash.h"
#include "Arquivo.h"
#include <sstream>
#include <stdlib.h>
#include "ArvoreAVL.h"
#include "ArvoreB.h"
#include <time.h>       /* time */
#include <random>

using namespace std;

int random(int min, int max) {

    unsigned long x;
    x = rand();
    x <<= 15;
    x ^= rand();
    x %= max;

    return x;
}


NodeHT* getRandomValue(HashTable* ht){
    srand(time(NULL));

    int i = 0, max = ht->getTam();
    do{
        i = random(0, max);

    } while(ht->operator[](i)==NULL);
    NodeHT* n= ht->operator[](i);
    return n;
}

int hashAleatorio(HashTable* ht){
    srand(time(NULL));
    int i = 0, max = ht->getTam();
    do{
        i = random(0, max);
    } while(ht->operator[](i)==NULL);
    return i;
}

void teste(int n,int op,HashTable* dados,ArvBinBusca*loc){

    ofstream filestream("filename.txt");

    switch (op) {
    case 1: {
        HashTable h(n + n / 20);

        for (int i = 0; i < n; i++) {

            NodeHT* m = (getRandomValue(dados));
            h.insere(m);
        }
        h.print((n < 20) ? cout : filestream);
        break;
    }

    case 2: {
        ArvoreAVL arv;

        for (int i = 0; i < n; i++) {
            NodeHT* m;
            do {
                m = (getRandomValue(dados));

            } while (arv.busca(m->codcidade));
            arv.insere(m->codcidade);
        }
        arv.imprime((n < 20) ? cout : filestream);

        break;
    }

    case 3: {
        ArvBinBusca arv;
        for (int i = 0; i < n; i++) {

            NodeHT* m;
            NoQArv* na;
            do {
                m = (getRandomValue(dados));
               na = loc->busca(m->codcidade);

            } while (arv.busca(m->codcidade)==NULL);

            arv.insere(na);
        }

        arv.imprime((n < 20) ? cout : filestream);
        break;
    }


    case 4: {
        ArvoreB arv(4);

        for (int i = 0; i < n; i++) {

            NodeHT* m;
          do {
                m = (getRandomValue(dados));

            } while (arv.busca(m->codcidade));
            arv.insere(m->codcidade);
        }
        arv.imprime((n < 20) ? cout : filestream);

        break;
    }
    }
}

void analiseBusca(int n, HashTable* h, int codCidade){
    ArvoreAVL avl(h); ArvoreB bMenor(20,h); ArvoreB bMaior(200,h);

        for(int t=0;t<5;t++){   // 5 diferentes conjuntos
            for (int i = 0; i < n; i++) {
            avl.insere(hashAleatorio(h));
            bMenor.insere(hashAleatorio(h));
            bMaior.insere(hashAleatorio(h));
        }
        //BUSCA S1
        cout<< avl.busca(codCidade);
        cout<< bMenor.busca(codCidade);
        cout<< bMenor.busca(codCidade);

        //FALTA BUSCAS2

        }

}



int main()
{
    Arquivo arq;
    HashTable* ht=new HashTable(200069);
    ArvBinBusca* Q=new ArvBinBusca ();
    arq.leArqProcessado(ht,"processado.csv",100000);
    arq.leArqCord(Q, "coordenadas.csv", -1);

    analiseBusca(500, ht, 509786);

   // teste(50, 1, ht, Q);
  //  teste(50, 2, ht, Q);
   // teste(50,4,ht,Q); */

    return 0;
}

