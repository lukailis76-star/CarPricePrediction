
# Car Price Prediction

## Opis
Projekat predvidja cijenu polovnih automobila u dolarima pomocu masinskog ucenja.

## Podaci
Koristi se fajl cars.csv. Ciljna kolona je priceUSD.

## Modeli
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Evaluacija
Modeli se uporedjuju pomocu MAE, RMSE i R2 metrika.

## Pokretanje
Instalirati biblioteke:

pip install -r requirements.txt

Pokrenuti projekat:

python main.py

## Sacuvani model
Najbolji model se cuva u fajlu car_price_model.joblib.
