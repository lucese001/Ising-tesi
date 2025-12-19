#include <stdio.h>
#include <stdlib.h>

// --------------------------------------------------------------------
// Converte indice → coordinate
// --------------------------------------------------------------------
void coord_from_index(size_t index, size_t n_dim, size_t *arr, size_t *coord) {
    for (size_t d = 0; d < n_dim; d++) {
        coord[d] = index % arr[d];
        index /= arr[d];
    }
}

// --------------------------------------------------------------------
// Converte coordinate → indice
// --------------------------------------------------------------------
size_t index_from_coord(size_t n_dim, size_t *arr, size_t *coord) {
    size_t index = 0;
    size_t mult = 1;

    for (size_t d = 0; d < n_dim; d++) {
        index += coord[d] * mult;
        mult *= arr[d];
    }

    return index;
}

// --------------------------------------------------------------------
// MAIN
// --------------------------------------------------------------------
int main(void) {

    FILE *fp = fopen("dimensioni.txt", "r");
    if (!fp) {
        perror("Errore apertura file");
        return 1;
    }

    size_t n_dim;
    fscanf(fp, "%zu", &n_dim);

    size_t *arr = malloc(n_dim * sizeof(size_t));
    for (size_t i = 0; i < n_dim; i++)
        fscanf(fp, "%zu", &arr[i]);

    fclose(fp);

    // ------------------------ TEST ------------------------

    // Coordinate di test
    size_t *coord = malloc(n_dim * sizeof(size_t));

    printf("Inserisci %zu coordinate:\n", n_dim);
    for (size_t d = 0; d < n_dim; d++)
        scanf("%zu", &coord[d]);

    // coordinate → indice
    size_t idx = index_from_coord(n_dim, arr, coord);
    printf("Indice lineare = %zu\n", idx);

    // indice → coordinate (ricontrollo)
    size_t *back = malloc(n_dim * sizeof(size_t));
    coord_from_index(idx, n_dim, arr, back);

    printf("Coordinate ricostruite: ");
    for (size_t d = 0; d < n_dim; d++)
        printf("%zu ", back[d]);
    printf("\n");

    free(arr);
    free(coord);
    free(back);

    return 0;
}
