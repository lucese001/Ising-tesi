#include "prng_engine.hpp"
#include <cstdint>
#include <random>
#include <vector>
#include <cstdio>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mpi.h>

using namespace std;

#define PARALLEL_RNG

/** Parameters (some remain global because used in many functions) */
const double Beta = 0.29;
size_t N = 0;              // totale numero di siti (prodotto delle dimensioni)
size_t nThreads = 1;       // numero di thread OpenMP
size_t N_dim = 0;          // numero di dimensioni
vector<size_t> arr;        // lunghezze per dimensione
const int seed = 124634;
int nConfs = 0;

struct timer
{
  static timer timerCost;

  static auto now() { return chrono::high_resolution_clock::now(); }

  size_t tot = 0;
  size_t n = 0;
  chrono::high_resolution_clock::time_point from;

  void start() { n++; from = now(); }
  void stop() { tot += chrono::duration_cast<chrono::nanoseconds>(now()-from).count(); }

  double get(bool sub = true) {
    double res = (double) tot;
    if (sub && timerCost.n > 0)
      res -= timerCost.tot * (double)n / (double)timerCost.n;
    return res / 1e9;
  }
};
timer timer::timerCost;

// --------------------------------------------------------------------
// index <-> coord (generalizzato n-d)
// --------------------------------------------------------------------
// index_to_coord scrive le coordinate in coord_buf (array di length n_dim)
// convenzione: coord[0] varia più veloce (mixed-radix come prima)
inline void index_to_coord(size_t index, size_t n_dim, const size_t *arr_ptr, size_t *coord_buf) {
    for (size_t d = 0; d < n_dim; ++d) {
        coord_buf[d] = index % arr_ptr[d];
        index /= arr_ptr[d];
    }
}

// coord_to_index legge coord_buf e restituisce index
inline size_t coord_to_index(size_t n_dim, const size_t *arr_ptr, const size_t *coord_buf) {
    size_t index = 0;
    size_t mult = 1;
    for (size_t d = 0; d < n_dim; ++d) {
        index += coord_buf[d] * mult;
        mult *= arr_ptr[d];
    }
    return index;
}

// --------------------------------------------------------------------
// computeEnSite: energia locale attorno a iSite (somma sulle 2*N_dim connessioni)
// - Non fa deduplicazione: è energia locale (utile per Metropolis).
// - Usa condizioni periodiche modulo arr[d].
// --------------------------------------------------------------------
int computeEnSite(const vector<int>& conf, const size_t& iSite) {
    // buffer temporanei per coordinate
    vector<size_t> coord_site(N_dim);
    vector<size_t> coord_neigh(N_dim);

    // ricava la coordinata n-d del sito iSite
    index_to_coord(iSite, N_dim, arr.data(), coord_site.data());

    int en = 0;
    // per ogni dimensione considera +1 e -1
    for (size_t d = 0; d < N_dim; ++d) {
        memcpy(coord_neigh.data(), coord_site.data(), N_dim * sizeof(size_t));
        // vicino +1
        coord_neigh[d] = (coord_site[d] + 1) % arr[d];
        size_t idx_plus = coord_to_index(N_dim, arr.data(), coord_neigh.data());
        en -= conf[idx_plus] * conf[iSite];

        // vicino -1
        memcpy(coord_neigh.data(), coord_site.data(), N_dim * sizeof(size_t));
        coord_neigh[d] = (coord_site[d] + arr[d] - 1) % arr[d];
        size_t idx_minus = coord_to_index(N_dim, arr.data(), coord_neigh.data());
        en -= conf[idx_minus] * conf[iSite];
    }

    return en;
}

// --------------------------------------------------------------------
// computeEn: energia totale (riduzione parallela)
// Nota: qui sommiamo le energie locali. Se vuoi la "classica" energia
//  con metà delle connessioni sommate una sola volta, puoi dividere il
//  risultato per 2 alla fine.
// --------------------------------------------------------------------
int computeEn(const vector<int>& conf) {
    long long en = 0; // usare tipo più largo per riduzione
#pragma omp parallel for reduction(+:en)
    for (size_t iSite = 0; iSite < N; ++iSite) {
        // per efficienza potresti voler contare solo direzioni positive
        // per non doppiare; ma lasciamo come locale completo (utile per dE locali)
        en += computeEnSite(conf, iSite);
    }
    return (int) en;
}

// --------------------------------------------------------------------
// computeMagnetization
// --------------------------------------------------------------------
double computeMagnetization(const vector<int>& conf) {
    long long mag = 0;
#pragma omp parallel for reduction(+:mag)
    for (size_t iSite = 0; iSite < N; ++iSite)
        mag += conf[iSite];
    return (double) mag / (double) N;
}
// --------------------------------------------------------------------
// halo_index: trova i rank vicini lungo ogni dimensione MPI
// --------------------------------------------------------------------
void halo_index(MPI_Comm cart_comm,
                int ndims,
                std::vector<std::vector<int>>& neighbors)
{
    neighbors.resize(ndims, std::vector<int>(2));

    for (int d = 0; d < ndims; ++d) {
        int rank_source, rank_dest;
        MPI_Cart_shift(cart_comm, d, 1, &rank_source, &rank_dest);
        neighbors[d][0] = rank_source;
        neighbors[d][1] = rank_dest;
    }
}


int main(int argc, char** argv) {
    /* Legge file con istruzioni del tipo:
       N_dim
       arr[0] arr[1] ... arr[N_dim-1]
       nConfs
       nThreads
    */
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    FILE *fp = fopen("dimensioni.txt", "r");
    if (!fp) {
        perror("Errore apertura file");
        return 1;
    }

    if (fscanf(fp, "%zu", &N_dim) != 1) {
        fprintf(stderr, "Errore: impossibile leggere N_dim\n");
        fclose(fp);
        return 1;
    }

    // legge lunghezze per dimensione
    arr.assign(N_dim, 0);
    for (size_t i = 0; i < N_dim; ++i) {
        if (fscanf(fp, "%zu", &arr[i]) != 1) {
            fprintf(stderr, "Errore: lettura arr[%zu]\n", i);
            fclose(fp);
            return 1;
        }
    }

    if (fscanf(fp, "%d", &nConfs) != 1) {
        fprintf(stderr, "Errore: lettura nConfs\n");
        fclose(fp);
        return 1;
    }
    if (fscanf(fp, "%zu", &nThreads) != 1) {
        // se manca é 1
        nThreads = 1;
    }

    fclose(fp);

    // calcola N = prodotto arr[i]
    N = 1;
    for (size_t i = 0; i < N_dim; ++i) {
        if (arr[i] == 0) {
            fprintf(stderr, "Errore: arr[%zu] = 0\n", i);
            return 1;
        }
        N *= arr[i];
    }
    
    vector<int> Chunks(N_dim);
    MPI_Dims_create(world_size, N_dim, Chunks);
    // --- topologia cartesiana MPI ---
vector<int> periods(N_dim, 1);  // condizioni periodiche
MPI_Comm cart_comm;
// --- vicini MPI ---
std::vector<std::vector<int>> neighbors;
halo_index(cart_comm, (int)N_dim, neighbors);


MPI_Cart_create(MPI_COMM_WORLD,
                (int)N_dim,
                Chunks.data(),
                periods.data(),
                1,
                &cart_comm);


    omp_set_num_threads((int)nThreads);
    timer totalTime;
    for (size_t i = 0; i < 100000; ++i) { timer::timerCost.start(); timer::timerCost.stop(); }

    printf("N_dim: %zu, Npunti: %zu, NThreads: %zu, nConfs: %d\n", N_dim, N, nThreads, nConfs);

    FILE* measFile = fopen("meas.txt", "w");
    if (!measFile) { perror("meas.txt"); return 1; }

    vector<int> conf(N);

#ifdef PARALLEL_RNG
    prng_engine gen(seed);
    printf("Memory usage of the rng: %zu Bytes\n", sizeof(prng_engine));
#else
    std::vector<mt19937_64> gen(N);
    for (size_t iSite = 0; iSite < N; ++iSite) gen[iSite].seed(seed + (int)iSite);
    printf("Memory usage of the rng: %zu MB\n", N * sizeof(mt19937_64) / (1 << 20));
#endif

    // crea prima configurazione
    for (size_t i = 0; i < N; ++i) {
#ifdef PARALLEL_RNG
        prng_engine& genView = gen;
#else
        mt19937_64& genView = gen[i];
#endif
        conf[i] = binomial_distribution<int>(1, 0.5)(genView) * 2 - 1;
    }

    totalTime.start();

    vector<size_t> coord_buf(N_dim);
    vector<size_t> coord_tmp(N_dim);

    for (int iConf = 0; iConf < nConfs; ++iConf) {
        // aggiornamento a scacchiera: par = 0/1
        for (int par = 0; par < 2; ++par) {
#ifdef PARALLEL_RNG
#pragma omp parallel
            {
                const size_t iThread = omp_get_thread_num();
                const size_t chunkSize = (N + nThreads - 1) / nThreads;
                const size_t beg = chunkSize * iThread;
                const size_t end = std::min(N, beg + chunkSize);
                prng_engine genView = gen;
                genView.discard(2 * 2 * (beg / 2 + N / 2 * (par + 2 * iConf)));
                for (size_t iSite = beg; iSite < end; ++iSite) {
#else
#pragma omp parallel for
                for (size_t iSite = 0; iSite < N; ++iSite) {
                    mt19937_64& genView = gen[iSite];
#endif
                    // compute parity pSite = sum(coord) % 2
                    index_to_coord(iSite, N_dim, arr.data(), coord_buf.data());
                    size_t sum = 0;
                    for (size_t d = 0; d < N_dim; ++d) sum += coord_buf[d];
                    size_t pSite = sum % 2;

                    if (par == (int)pSite) {
                        const int oldVal = conf[iSite];
                        const int enBefore = computeEnSite(conf, iSite);

                        conf[iSite] = binomial_distribution<int>(1, 0.5)(genView) * 2 - 1;

                        const int enAfter = computeEnSite(conf, iSite);
                        const int eDiff = enAfter - enBefore;
                        const double pAcc = std::min(1.0, exp(-Beta * (double)eDiff));
                        const int acc = binomial_distribution<int>(1, pAcc)(genView);

                        if (not acc) conf[iSite] = oldVal;
                    }
#ifdef PARALLEL_RNG
                } // end for iSite chunk
            } // end parallel
#else
                } // end omp parallel for
#endif
        } // end for par
        MPI_Barrier(MPI_COMM_WORLD);
        for (size_t d = 0; d < N_dim; ++d) {
            printf("Rank %d | dim %zu : - -> %d , + -> %d\n",
                   world_rank, d,
                   neighbors[d][0],
                   neighbors[d][1]);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        const double mag = computeMagnetization(conf);
        const double ene = computeEn(conf);

        fprintf(measFile, "%lg %lg\n", mag, ene);
        fflush(measFile);

        printf("Progress: %d/%d\n", iConf+1, nConfs);
    } // end nConfs

    fclose(measFile);
    totalTime.stop();
    printf("Duration: %lg s\n", totalTime.get());
    MPI_Comm_free(&cart_comm);
MPI_Finalize();

    return 0;
}