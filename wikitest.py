import wikipedia
from wikipedia import WikipediaPage

wiki = WikipediaPage("Indiana Jones and the Raiders of the Lost Ark")

print("Ark" in wiki.links)
