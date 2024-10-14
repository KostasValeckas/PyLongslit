import cdspyreadme

tablemaker = cdspyreadme.CDSTablesMaker()

# add a table
table = tablemaker.addTable("gtcspec.dat", description="my ascii table")
# write table in CDS-ASCII aligned format (required)
tablemaker.writeCDSTables()

# Customize ReadMe output
tablemaker.title = "The galaxy counterpart and environment of the dusty Damped Lyman-alpha Absorber at z=2.226 towards Q1218+0832"
tablemaker.author = 'J.P.U.Fynbo, L.B.Christensen, S.J.Geier, K.E.Heintz, J.-K.Krogager, C.Ledoux, B.Milvang-Jensen, P.Moller, S.Vejlgaard, J.Viuho, G.Ostlin'
tablemaker.date = 2023
#tablemaker.abstract = ""
#tablemaker.more_description = "Additional information of the data context."
#tablemaker.putRef("II/246", "2mass catalogue")
#tablemaker.putRef("http://...", "external link")

# Print ReadMe
tablemaker.makeReadMe()

# Save ReadMe into a file
with open("ReadMe", "w") as fd:
    tablemaker.makeReadMe(out=fd)
