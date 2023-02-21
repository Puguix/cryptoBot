import discord

TOKEN = ""

client = discord.Client(intents=discord.Intents.all())


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')
    
    global general
    general = client.get_channel()
    
    global botChannel
    botChannel = client.get_channel()
    
    global simon
    simon = await client.fetch_user()

    global paul
    paul = await client.fetch_user()
    
    await botChannel.send(":green_circle: \n ca marche")


@client.event
async def on_message(message):
        
    if message.author == simon:
        message.channel.send("Ta gueule Simon")

client.run(TOKEN)

# await client.close()