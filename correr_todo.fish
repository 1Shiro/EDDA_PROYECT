#!/usr/bin/env fish

# Siempre desde la carpeta del proyecto (los scripts usan rutas relativas)
cd (dirname (status filename))

# Los gráficos se guardan en archivos, sin abrir ventanas
set -x MPLBACKEND Agg

mkdir -p resultados/plots


echo " 𓆉 Revisando Python y matplotlib 𓆉 "
python3 -c 'import sys; assert sys.version_info >= (3, 8), "Se necesita Python 3.8+"'; or exit 1
python3 -c 'import matplotlib' 2>/dev/null; or begin
    echo "  falta matplotlib: python3 -m pip install -r requirements.txt"
    exit 1
end
echo "  "(python3 --version)", matplotlib "(python3 -c 'import matplotlib; print(matplotlib.__version__)')


echo " 𓆉 Tests 𓆉 "
for t in test_algorithms test_incremental
    echo "  -> $t"
    python3 $t.py; or exit 1
end


echo " 𓆉 Exp 1: Jarvis, Andrew y Quickhull (~12 min) 𓆉 "
python3 benchmark.py; or exit 1
python3 plot_results.py; or exit 1


echo " 𓆉 Exp 2: inserción contra recalcular (~2 min) 𓆉 "
python3 compare_incremental.py; or exit 1


echo " 𓆉 Listo 𓆉 CSV y gráficos en resultados/ 𓆉 "
