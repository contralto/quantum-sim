#=
qc:
- Julia version: 1.5.3
- Author: cgon (contralto)
- Date: 2020-12-23
=#


# TODO: show_graph, plot_state, pixel_state, complex_to_rgb, hsv_to_rgb

using LinearAlgebra

h = [[1/sqrt(2), 1/sqrt(2)],
    [1/sqrt(2), -1/sqrt(2)]]

function pair_exchange!(state, gate, m0, m1)
    x = state[m0]
    y = state[m1]
    state[m0] = gate[1][1] * x + gate[1][2] * y
    state[m1] = gate[2][1] * x + gate[2][2] * y
end


function transform!(state, target, gate, cond::Function = f(x::Int)::Bool = true)
    # sections begin when we reach a new set of "first qbits" (i.e. first in a pair)
    # ex. with distance 2 and state [1, 0, 0, 0, 0, 0, 0, 0], the sections would be
    # [1*, 0*, 0, 0], [0*, 0*, 0, 0] (* signifies the first qbit of each pair)

    # number of qbits in each section of the state
    section_len = Int(2 ^ (target + 1))

    # distance between paired qbits
    distance = Int(section_len / 2)

    # applies the gate to each pair
    for section in 0:(length(state) / section_len - 1)
        for pair in 0:(section_len / 2 - 1)
            # index of first qbit in pair
            m0 = Int(section_len * (section) + pair)
            if (cond(m0))
                # index of second qbit in pair
                m1 = m0 + distance
                pair_exchange!(state, gate, m0 + 1, m1 + 1)
            end
        end
    end
end




function transform_with_matrix!(state, t, gate, cond::Function = f(x::Int)::Bool = true)
    n = length(state)
    G = Matrix{Float64}(I, n, n)

    factor = 2 ^ (t + 1)
    shift = Int(2 ^ t)

    for prefix in 0:(n / factor - 1)
        for suffix in 0:(2^t - 1)
            m0 = Int(factor * (prefix) + suffix)
            if (cond(m0))
                # + 1 for julia indexing
                m0 += 1
                m1 = m0 + shift
                G[m0, m0] = gate[1][1]
                G[m0, m1] = gate[1][2]
                G[m1, m0] = gate[2][1]
                G[m1, m1] = gate[2][2]
            end
        end
    end
#     @show G * state
    show(stdout, "text/plain", G)
    println()
    println()
    state .= G * state
end


# initializes the state as an array of size 2^n
# index 1 has proportion 1, the rest have nothing (proportion 0)
function init_state(n::Int)::Array{Float64, 1}
    state::Array{Float64, 1} = [0 for _ in 1:2 ^ n]
    state[1] = 1
    return state
end

f(x::Int) = x % 2 == 0

function test_play()
    state = init_state(3)
    for t in 0:2
        transform_with_matrix!(state, t, h)
    end
#     println(state)
@show state
end

test_play()
