#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <string.h>

int read_matrix_from_file(const char* filename, double* matrix, int n) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Cannot open matrix file");
        return 0;
    }

    int size_from_file;
    if (fscanf(file, "%d", &size_from_file) != 1 || size_from_file != n) {
        fprintf(stderr, "Matrix size in file doesn't match the provided size.\n");
        fclose(file);
        return 0;
    }

    for (int i = 0; i < n * (n + 1); i++) {
        if (fscanf(file, "%lf", &matrix[i]) != 1) {
            fprintf(stderr, "Error reading matrix data.\n");
            fclose(file);
            return 0;
        }
    }

    fclose(file);
    return 1;
}

int main(int argc, char* argv[]) {
    int rank, size, n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        FILE* file = fopen("matrix.txt", "r");
        if (!file) {
            perror("Cannot open matrix file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (fscanf(file, "%d", &n) != 1) {
            fprintf(stderr, "Failed to read matrix size from file.\n");
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fclose(file);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // перевірка: кількість процесів не повинна перевищувати n
    if (size > n) {
        if (rank == 0)
            fprintf(stderr, "Number of processes (%d) must be ? matrix size (%d).\n", size, n);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int base_rows = n / size;
    int extra = n % size;
    int local_rows = base_rows + (rank < extra ? 1 : 0);
    int start_row = rank * base_rows + (rank < extra ? rank : extra);

    double* local_a = (double*)malloc(local_rows * (n + 1) * sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));

    if (rank == 0) {
        double* full_matrix = (double*)malloc(n * (n + 1) * sizeof(double));

        if (!read_matrix_from_file("matrix.txt", full_matrix, n)) {
            free(full_matrix);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int offset = 0;
        for (int p = 0; p < size; p++) {
            int rows = base_rows + (p < extra ? 1 : 0);
            if (p == 0)
                memcpy(local_a, full_matrix, rows * (n + 1) * sizeof(double));
            else
                MPI_Send(full_matrix + offset * (n + 1), rows * (n + 1), MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            offset += rows;
        }
        free(full_matrix);
    }
    else {
        MPI_Recv(local_a, local_rows * (n + 1), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    double* pivot_row = (double*)malloc((n + 1) * sizeof(double));

    int* row_owner = (int*)malloc(n * sizeof(int));
    for (int i = 0, offset = 0; i < size; i++) {
        int rows = base_rows + (i < extra ? 1 : 0);
        for (int j = 0; j < rows; j++) {
            row_owner[offset++] = i;
        }
    }

    //прямий хід
    for (int k = 0; k < n - 1; k++) {
        int owner = row_owner[k];

        if (rank == owner) {
            int local_index = k - start_row;
            if (local_index >= 0 && local_index < local_rows) {
                memcpy(pivot_row, &local_a[local_index * (n + 1)], (n + 1) * sizeof(double));
            }
            else {
                memset(pivot_row, 0, (n + 1) * sizeof(double));
            }
        }

        MPI_Bcast(pivot_row, n + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        for (int i = 0; i < local_rows; i++) {
            int global_i = start_row + i;
            if (global_i > k) {
                if (pivot_row[k] == 0.0) continue;
                double factor = local_a[i * (n + 1) + k] / pivot_row[k];
                for (int j = k; j <= n; j++) {
                    local_a[i * (n + 1) + j] -= factor * pivot_row[j];
                }
            }
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);
    double end;

    double* full_a = NULL;
    if (rank == 0)
        full_a = (double*)malloc(n * (n + 1) * sizeof(double));

    int* recv_counts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        recv_counts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        for (int i = 0, offset = 0; i < size; i++) {
            int rows = base_rows + (i < extra ? 1 : 0);
            recv_counts[i] = rows * (n + 1);
            displs[i] = offset;
            offset += recv_counts[i];
        }
    }

    MPI_Gatherv(local_a, local_rows * (n + 1), MPI_DOUBLE,
        full_a, recv_counts, displs, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        //зворотній хід
        for (int i = n - 1; i >= 0; i--) {
            x[i] = full_a[i * (n + 1) + n];
            for (int j = i + 1; j < n; j++)
                x[i] -= full_a[i * (n + 1) + j] * x[j];
            x[i] /= full_a[i * (n + 1) + i];
        }

        end = MPI_Wtime();

        printf("\nSolution:\n");
        for (int i = 0; i < n; i++)
            printf("x%d = %.6lf\n", i + 1, x[i]);
        printf("\nTime taken: %.6f seconds\n", end - start);

        free(full_a);
        free(recv_counts);
        free(displs);
    }

    free(local_a);
    free(pivot_row);
    free(x);

    MPI_Finalize();
    return 0;
}