from setuptools import find_packages, setup


setup(
    name="alopecia-federated-prognosis",
    version="0.1.0",
    description="Research workspace for federated multimodal alopecia prognosis.",
    author="Timothy Musharu",
    packages=find_packages(
        include=[
            "data",
            "data.*",
            "models",
            "models.*",
            "evaluation",
            "evaluation.*",
            "experiments",
            "experiments.*",
        ]
    ),
    include_package_data=True,
    install_requires=[
        "PyYAML==6.0.2",
    ],
)
