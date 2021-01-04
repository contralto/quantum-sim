#=
qc:
- Julia version: 1.5.3
- Author: cgon (contralto)
- Date: 2020-12-23
=#

using LinearAlgebra

global counter = 0
global folder = "IdeaProjects/quantum-sim/images/"

const Amplitude = Complex

const State = Vector{Amplitude}

struct Gate
    name::String
    data::Vector{Vector{Amplitude}}
end

############################### GATES ##########################################

h = Gate("H", [[Complex(1/sqrt(2)), 1/sqrt(2)],
                       [1/sqrt(2), -1/sqrt(2)]])

function phase(theta)
    return Gate("Phase(" * string(round(theta,digits=3)) * ")",
                [[1, 0], [0, (cos(theta) + im * sin(theta))]])
end

############################### ENCODING ##########################################

# # check cth digit of binary string of x is 1
function isDigitOne(x::Int, c::Int, n::Int)
    str = reverse(string(x, base = 2, pad = n))
    return str[c + 1] == '1'
end

function pair_exchange!(state::State, gate::Gate, m0::Int, m1::Int)
    x = state[m0]
    y = state[m1]
    state[m0] = gate.data[1][1] * x + gate.data[1][2] * y
    state[m1] = gate.data[2][1] * x + gate.data[2][2] * y
end

function c_transform(state::State, c::Int, t::Int, gate::Gate)
    c = Int(log2(length(state))) - c - 1
    transform!(state, t, gate, m -> isDigitOne(m, c, Int(log2(length(state)))))
end

function iqft(state::State, targets::Array)
    for j in reverse(targets)
        transform!(state, j, h)
        for k in reverse(0:j - 1)
            c_transform(state, j, targets[k + 1], phase(-pi / (2 ^ (j-k))))
        end
    end
end


function param_encoding(state, targets, v)
    theta = v * 2 * pi / (2 ^ length(targets))

@show state
    for j in targets
        transform!(state, j, h)
    end

    for i in targets
        transform!(state, i, phase(2 ^ i * theta))
    end

    iqft(state, targets)
end


############################### TRANSFORMATIONS ##########################################

function transform!(state::State, target::Int, gate::Gate,
                    cond::Function = f(x::Int)::Bool = true)
    # sections begin when we reach a new set of "first qubits" (i.e. first in a pair)
    # ex. with distance 2 and state [1, 0, 0, 0, 0, 0, 0, 0], the sections would be
    # [1*, 0*, 0, 0], [0*, 0*, 0, 0] (* signifies the first qubit of each pair)

    target = Int(log2(length(state))) - target - 1

    # number of qubits in each section of the state
    section_len = Int(2 ^ (target + 1))

    # distance between paired qubits
    distance = Int(section_len / 2)

    # applies the gate to each pair
    for section in 0:(length(state) / section_len - 1)
        for pair in 0:(section_len / 2 - 1)
            # index of first qubit in pair
            m0 = Int(section_len * (section) + pair)
            if (cond(m0))
                # index of second qubit in pair
                m1 = m0 + distance
                pair_exchange!(state, gate, m0 + 1, m1 + 1)
            end
        end
    end
    title = "Gate: " * gate.name * ", target: " * string(target)
    save_state(state, title)
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
                G[m0, m0] = gate.data[1][1]
                G[m0, m1] = gate.data[1][2]
                G[m1, m0] = gate.data[2][1]
                G[m1, m1] = gate.data[2][2]
            end
        end
    end
    state .= G * state
end

############################### GRAPHING ##########################################
using Gadfly, Colors

# https://gist.github.com/eyecatchup/9536706 Colors
function complex_to_hsv(z::Complex, change_sat::Bool = true)
    r = real(z)
    i = imag(z)

    val = 1.0

    hue = angle(z) * 180 / pi
    if (hue < 0) hue += 360 end

    if (change_sat) sat = abs(z)
    else sat = 1.0 end

    # Make sure our arguments stay in-range
    hue = max(0, min(360, hue))
    sat = max(0.0, min(1.0, sat))
    val = max(0, min(100, val))

    return (hue, sat, val)
end

function bar_state(state, amp::Bool = true, title::String = "default")
    l = length(state)
    outcomes = collect(0:l)
    #amplitudes (colors based on vector)
    if (amp)
        data = round.(abs.(state), digits = 3)
        bar_colors = [HSV(h, 1, v) for (h, _, v) in complex_to_hsv.(Complex.(state))]
        ylabel = "Amplitude"
#         if (title == "default")
#             title = "Outcome Amplitudes"
#         end
    #probabilities (all red)
    else
        data =round.(abs.(state).^2, digits = 3)
        bar_colors = [HSV(0,1,1) for a in state]
        ylabel = "Probability"
#         if (title == "default")
#             title = "Outcome Probabilities"
#         end
    end


    tufte_bar = Theme(
        default_color = colorant"green",
        background_color = colorant"white",
        bar_spacing = 2pt,
        grid_line_width = 0.25pt,
        minor_label_font = "Gill Sans",
        major_label_font = "Gill Sans",
        alphas = [0.7], key_position=:none
    )

    set_default_plot_size(max(10, l) * cm, 10 * cm)
    plot(x = [string(i) * " = " * string(i, base=2, pad=Int(log2(l))) for i in 0:(l - 1)],
         y = data, label=string.(data),
         Guide.xticks(ticks = outcomes, orientation = :vertical),
         Guide.yticks(ticks=[i * 0.1 for i = 0:10]),
         Coord.cartesian(ymin = 0.0, ymax = 1.1),
         Geom.bar, Geom.label(position=:above),
         Guide.xlabel("Outcomes"),
         Guide.ylabel(ylabel),
         Guide.title(title),
         color=0:l-1,
         Scale.color_discrete_manual(bar_colors...),
         tufte_bar)
end

function pixel_state(state::State, title::String = "Outcome Amplitudes")
    l = size(state)[1]
    outcomes = [0:l;]
    pixel_colors = [HSV(h, s, v) for (h, s, v) in complex_to_hsv.(Complex.(state))]

    tufte_bar = Theme(
        default_color=colorant"gray",
        background_color=colorant"white",
        bar_spacing=1pt,
        grid_line_width=0pt,
        minor_label_font="Gill Sans",
        major_label_font="Gill Sans",
        key_position=:none
    )

    set_default_plot_size(4cm, max(10, l)*cm)
    probabilities = round.(abs.(state).^2, digits=3)

    plot(y=[string(i)*"="*string(i, base=2, pad=Int(log2(l))) for i in (0:l-1)], x=[0.05 for _=0:l-1],
        Guide.ylabel(""), Guide.yticks(ticks=outcomes),
        Guide.xlabel(""), Guide.xticks(ticks=nothing),
        Coord.cartesian(xmin=0.0, xmax=0.1),
        Geom.bar(orientation=:horizontal),
        Guide.title(title),
        tufte_bar,
        color=0:l-1, Scale.color_discrete_manual(pixel_colors...))
end

import Cairo, Fontconfig

function save_state(state::State, title::String = "default", amplitude::Bool = true, bar::Bool = true)
    if (bar)
        draw(PNG(joinpath(homedir(), folder * string(counter) * ".png"), max(10, length(state)) * cm, 10cm), bar_state(state, amplitude, title))
    else
        draw(PNG(joinpath(homedir(), folder * string(counter) * ".png"), max(10, length(state)) * cm, 10cm), pixel_state(state, title))
    end
    global counter += 1
end


############################### INITIALIZATION ##########################################

# initializes the state as an array of size 2^n
# index 1 has proportion 1, the rest have nothing (proportion 0)
function init_state(n::Int)::Array{Amplitude, 1}
    state::State = [0 for _ in 1:2 ^ n]
    state[1] = 1
    return state
end

############################### RUN METHODS ##########################################

function test_transform()
    n = 4
    state = init_state(n)
    save_state(state, "Initial State")

    for t in 0:(n - 1)
        transform!(state, t, h)
    end
end

function test_param()
    n = 4
    state = init_state(n)
    save_state(state, "Initial State")

    param_encoding(state, collect(0:n - 1), 15)
    save_state(state, "After encoding " * string(15) * " to targets")

end

function test_phase()
    n = 4
    state = init_state(n)
    save_state(state, "Initial State")
    for t in 0:n-1
        transform!(state, t, h)
        save_state(state, "Phase of π/4 Applied to Qubit 1")
        transform!(state, t, phase(pi/4))
        save_state(state, "Phase of π/4 Applied to Qubit 2")
    end
end

test_transform()
test_param()
test_phase()