#include <stdio.h>
#include <stdlib.h>

int main(void) {
    FILE *fp = fopen("dimensioni.txt", "r");
    if (!fp) {
        perror("Errore apertura file");
        return 1;
    }

    size_t n_dim;
    size_t N=1;
    if (fscanf(fp, "%zu", &n_dim) != 1) {
        fprintf(stderr, "Errore: impossibile leggere la dimensione.\n");
        fclose(fp);
        return 1;
    }

    size_t *arr = malloc(n_dim * sizeof(size_t));
    size_t *step = malloc(n_dim * sizeof(size_t));
    if (!arr) {
        perror("Errore malloc");
        fclose(fp);
        return 1;
    }

    for (size_t i = 0; i < n_dim; i++) {
        if (fscanf(fp, "%zu", &arr[i]) != 1) {
            fprintf(stderr, "Errore: dati insufficienti nel file.\n");
            free(arr);
            fclose(fp);
            return 1;
        }
        N=N*arr[i];
        step[i]=N;
    }

    fclose(fp);

    // Stampa dell'array
    printf("Array letto dal file:\n");
    for (size_t i = 0; i < n_dim; i++) {
        printf("%zu ", arr[i]);
    }
    printf("step:\n");
    for (size_t i = 0; i < n_dim; i++) {
        printf("%zu ", step[i]);
    }

    free(arr);
    free(step);
    return 0;
    
}

