#ifndef COORDENADA_H_INCLUDED
#define COORDENADA_H_INCLUDED

struct coord{
    int x=NULL;
    int y=NULL;

    bool operator ==(coord b){
    return(x==b.x&&y==b.y);
    }
    void operator =(coord b){
        x=b.x;
        y=b.y;
    }
};
typedef struct coord coord;


#endif // COORDENADA_H_INCLUDED
