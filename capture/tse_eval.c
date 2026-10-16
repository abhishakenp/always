// tse_eval WEIGHTS VOICEPRINT.f32 E.wav R.wav OUT.wav : run tse_core offline (parity + CPU cost).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tse_core.h"
static short *rdw(const char *p, long *n){FILE*f=fopen(p,"rb");if(!f){perror(p);exit(2);}char id[4];unsigned sz;fread(id,1,4,f);fread(&sz,4,1,f);fread(id,1,4,f);
 for(;;){if(fread(id,1,4,f)!=4)exit(3);fread(&sz,4,1,f);if(!memcmp(id,"data",4))break;fseek(f,sz,SEEK_CUR);} short*d=malloc(sz);*n=fread(d,2,sz/2,f);fclose(f);return d;}
static void wrw(const char*p,short*d,long n){FILE*f=fopen(p,"wb");unsigned data=n*2,riff=36+data,fmt=16,rate=16000,br=32000;unsigned short pcm=1,ch=1,ba=2,bits=16;
 fwrite("RIFF",1,4,f);fwrite(&riff,4,1,f);fwrite("WAVEfmt ",1,8,f);fwrite(&fmt,4,1,f);fwrite(&pcm,2,1,f);fwrite(&ch,2,1,f);fwrite(&rate,4,1,f);fwrite(&br,4,1,f);fwrite(&ba,2,1,f);fwrite(&bits,2,1,f);fwrite("data",1,4,f);fwrite(&data,4,1,f);fwrite(d,2,n,f);fclose(f);}
int main(int c, char **v) {
    if (c < 6) { fprintf(stderr, "usage: tse_eval W VP.f32 E.wav R.wav OUT.wav\n"); return 1; }
    float vp[256]; FILE *f = fopen(v[2], "rb"); if (!f || fread(vp, 4, 256, f) != 256) return 2; fclose(f);
    tse_core *t = tse_new(v[1], vp, 256); if (!t) { fprintf(stderr, "load failed\n"); return 3; }
    long ne, nr; short *e = rdw(v[3], &ne), *r = rdw(v[4], &nr); long n = ne < nr ? ne : nr; n = n / 160 * 160;
    short *o = calloc(n, 2); float ef[160], rf[160], of[160];
    clock_t c0 = clock();
    for (long i = 0; i < n; i += 160) {
        for (int k = 0; k < 160; k++) { ef[k] = e[i + k] / 32768.f; rf[k] = r[i + k] / 32768.f; }
        tse_process(t, ef, rf, of);
        for (int k = 0; k < 160; k++) { float s = of[k] * 32768.f; o[i + k] = s > 32767 ? 32767 : s < -32768 ? -32768 : (short)s; }
    }
    double cpu = (double)(clock() - c0) / CLOCKS_PER_SEC;
    fprintf(stderr, "tse: %.1f s audio, %.3f s CPU = %.2f%% of one core, %.1f us/frame\n", n / 16000.0, cpu, 100 * cpu / (n / 16000.0), 1e6 * cpu / (n / 160));
    wrw(v[5], o, n); tse_free(t); return 0;
}
