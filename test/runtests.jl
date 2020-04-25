using Test

@time using YPPL_Diagnosis: mcmc_summary

@testset "YPPL_Diagnosis" begin
    post = randn(4, 500, 10)
    df = mcmc_summary(post)
    println(df)
    @test Bool.(df.Rhat .< 1.1) |> all
end

