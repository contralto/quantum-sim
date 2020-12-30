#=
qc:
- Julia version: 1.5.3
- Author: cgon (contralto)
- Date: 2020-12-23
=#


# TODO: show_graph, plot_state, pixel_state, complex_to_rgb, hsv_to_rgb

using LinearAlgebra, Gadfly

const Amplitude = ComplexF64
const State = Vector{Amplitude}
const Gate = Array{Array{Amplitude, 2}, 1}

h = [[ComplexF64(1/sqrt(2)) 1/sqrt(2)],
     [1/sqrt(2)            -1/sqrt(2)]]

function phase(theta)
    return [[1 0], [0 (cos(theta) + im * sin(theta))]]
end

function param_encoding(state, targets, v)
    theta = v * 2 * pi / (2 ^ length(targets))

    for j in targets
        transform!(state, j, h)
    end

    for i in targets
        transform!(state, i, phase(2 ^ (1im * theta)))
    end

    # iqft
end

function pair_exchange!(state::State, gate::Gate, m0::Int, m1::Int)
    x = state[m0]
    y = state[m1]
    state[m0] = gate[1][1] * x + gate[1][2] * y
    state[m1] = gate[2][1] * x + gate[2][2] * y
end

function transform!(state::State, target::Int, gate::Gate,
                    cond::Function = f(x::Int)::Bool = true)
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




function transform_with_matrix!(state::State, target::Int, gate::Gate,
                                cond::Function = f(x::Int)::Bool = true)
    n = length(state)
    G = Matrix{ComplexF64}(I, n, n)

    factor = 2 ^ (target + 1)
    shift =  2 ^ (target)

    for prefix in 0:(n / factor - 1)
        for suffix in 0:(2^target - 1)
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
#     show(stdout, "text/plain", G)
#     println()
#     println()
    state .= G * state
end

function plot_state(state)
    l = length(state)
    outcomes = collect(0:l)
    probabilities = round.(abs.(state).^2, digits = 3)

    tufte_bar = Theme(
        default_color = colorant"green",
        background_color = colorant"white",
        bar_spacing = 2pt,
        grid_line_width = 0.25pt,
        minor_label_font = "Gill Sans",
        major_label_font = "Gill Sans",
        alphas = [0.7],
    )

    set_default_plot_size(max(10, l) * cm, 10 * cm)

    plot(x = [string(i) * " = " * string(i, base=2, pad=Int(log2(l))) for i in 0:(l - 1)],
         y = probabilities, label=string.(probabilities),
         Guide.xticks(ticks = outcomes, orientation = :vertical),
         Guide.yticks(ticks=[i * 0.1 for i = 0:10]),
         Coord.cartesian(ymin = 0.0, ymax = 1.1),
         Geom.bar, Geom.label(position=:above),
         Guide.xlabel("Outcomes"),
         Guide.ylabel("Probability"),
         Guide.title("Outcome Probabilities"), tufte_bar)
end


# initializes the state as an array of size 2^n
# index 1 has proportion 1, the rest have nothing (proportion 0)
function init_state(n::Int)::Array{Amplitude, 1}
    state::State = [0 for _ in 1:2 ^ n]
    state[1] = 1
    return state
end

f(x::Int) = x % 2 == 0

function test_play()
    n = 3
    state = init_state(n)
    for t in 0:n - 1
        transform_with_matrix!(state, t, h)
    end
    @show state
    param_encoding(state, 1, 1)
    plot_state(state)
end

test_play()
