#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
  using ExecSpace = Kokkos::SYCL;

  const float step = (end - start) / static_cast<float>(count);

  struct SumPair {
    float sin_sum;
    float cos_sum;
  };

  SumPair result{0.0f, 0.0f};

  const int team_size = 128;
  const int vector_length = 4;

  const int league_size =
      (count + team_size * vector_length - 1) / (team_size * vector_length);

  Kokkos::parallel_reduce(
      "IntegralOptimized",
      Kokkos::TeamPolicy<ExecSpace>(league_size, team_size, vector_length),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team,
                    SumPair& global_sum) {
        SumPair local_sum{0.0f, 0.0f};

        const int base =
            team.league_rank() * team.team_size() * team.vector_length();

        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team,
                                    team.team_size() * team.vector_length()),
            [&](const int i, SumPair& thread_sum) {
              int idx = base + i;
              if (idx < count) {
                float x = start + (idx + 0.5f) * step;

                float s = sinf(x);
                float c = cosf(x);

                thread_sum.sin_sum += s;
                thread_sum.cos_sum += c;
              }
            },
            local_sum);

        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          global_sum.sin_sum += local_sum.sin_sum;
          global_sum.cos_sum += local_sum.cos_sum;
        });
      },
      result);

  return (result.sin_sum * step) * (result.cos_sum * step);
}