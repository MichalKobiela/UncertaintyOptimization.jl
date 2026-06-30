using UncertaintyOptimization
using Documenter

DocMeta.setdocmeta!(UncertaintyOptimization, :DocTestSetup, :(using UncertaintyOptimization); recursive=true)

makedocs(;
    modules=[UncertaintyOptimization],
    authors="Michal Kobiela <michal@walls.com.pl>, Mateusz K. Bieniek, Emma Pead and contributors",
    sitename="UncertaintyOptimization.jl",
    format=Documenter.HTML(;
        canonical="https://MichalKobiela.github.io/UncertaintyOptimization.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Workflow" => "workflow.md",
        "YAML Models" => "yaml.md",
        "Reference" => "reference.md"
    ],
)

deploydocs(;
    repo="github.com/MichalKobiela/UncertaintyOptimization.jl",
    devbranch="main",
)
