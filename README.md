# MovieLab · aplicació d’escriptori

MovieLab és un programa Python d’escriptori per experimentar amb sistemes de recomanació de pel·lícules. La interfície està construïda amb Tkinter i s’obre com una finestra normal del sistema.

## Algoritmes

1. Model base amb mitjana global i biaixos regularitzats.
2. K-NN usuari-usuari.
3. K-NN ítem-ítem.
4. Model basat en contingut mitjançant TF-IDF de gèneres i, opcionalment, títols.
5. Model híbrid que combina les quatre prediccions anteriors.

## Funcions de l’aplicació

- Base SQLite de demostració amb 60 pel·lícules, 90 usuaris i 1.842 valoracions.
- Obertura de carpetes MovieLens, parelles de CSV i bases SQLite.
- Desament del catàleg actiu com una nova base SQLite.
- Paràmetres independents per a tots dos K-NN: veïns, similitud mínima i regularització.
- Regularització i iteracions configurables per al model base.
- Inclusió opcional del títol i pes dels gèneres en el model de contingut.
- Pesos editables i normalitzats automàticament per al model híbrid.
- Optimitzador automàtic dels pesos híbrids segons RMSE o MAE.
- Comparació mitjançant RMSE, MAE, precisió@K, recall@K, NDCG@K, cobertura, diversitat, novetat i temps.
- Gràfic intercanviable per a qualsevol de les mètriques.
- Recomanacions filtrades per gènere, interval d’anys i nombre mínim de vots.
- Control per equilibrar afinitat i diversitat de la llista.
- Explicació de les recomanacions i de l’aportació de cada component híbrid.
- Exportació de comparacions i recomanacions a CSV.
- Cercador estadístic de pel·lícules per títol, gènere, vots i nota.
- Anàlisi individual d’usuaris, distribució de notes i gènere preferit.
- Comparació de gèneres per catàleg, valoracions, usuaris i nota mitjana.
- Informe de qualitat: dades absents, duplicats, esparsitat, casos amb pocs vots i desigualtat de popularitat.
- Corba de concentració per analitzar la cua llarga del catàleg.
- Formulari per crear un usuari, registrar-ne les valoracions i els gèneres preferits.
- Dos formularis d’alta: un de senzill per a l’ús quotidià i un altre d’avançat per experimentar.
- Recomanació immediata per a usuaris nous i opció de desar-los a la base activa.

## Iniciar

Des de la carpeta del projecte:

```bash
python3 -m pip install -r requirements.txt
python3 app.py
```

També es pot iniciar amb:

```bash
python3 desktop_app.py
```

## Carregar MovieLens

A la pestanya **Dades** o al menú **Fitxer**, selecciona una carpeta que contingui:

- `movies.csv`: `movieId`, `title`, `genres`;
- `ratings.csv`: `userId`, `movieId`, `rating` i, opcionalment, `timestamp`.

També es poden triar els dos fitxers per separat. Quan el conjunt supera les 2.500 pel·lícules, MovieLab ofereix limitar-lo als elements i usuaris més actius per evitar que les matrius K-NN consumeixin massa memòria.

## Comparació correcta

Tots els algoritmes utilitzen exactament la mateixa partició d’entrenament i prova. La partició es fa per usuari i deixa almenys dues valoracions per entrenar. Es poden modificar la proporció, la llavor, la nota considerada rellevant i la mida del top-K.

L’optimitzador de pesos utilitza la partició reservada com a validació. En un experiment acadèmic convé separar, a més, una prova final que no s’utilitzi durant l’optimització.

## Introduir un usuari nou

Obre la pestanya **Nou usuari** i tria un dels dos modes:

- **Senzill:** introdueix un nom, marca gèneres, tria almenys tres pel·lícules i respon amb “No em va agradar”, “Regular”, “Em va agradar” o “Em va encantar”. MovieLab configura automàticament el model híbrid.
- **Avançat:** permet introduir notes de 0,5 a 5 i controlar l’algoritme, la quantitat, l’impuls dels gèneres i la diversitat.

**Recomanar** crea un perfil temporal i entrena l’algoritme triat. **Desar usuari** incorpora les seves valoracions a la base activa. Per conservar-lo després de tancar el programa, utilitza **Fitxer → Desar base com a SQLite**.

## Proves

```bash
python3 -m unittest discover -s tests -v
```
