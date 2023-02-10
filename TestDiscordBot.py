import discord

TOKEN = "MTAzOTIyOTc5Njk2Nzc4ODY4NA.GEBYR2.bQWmWep-AqaeP8YKBZoWNzVp8AlXZf6z089BRk"

client = discord.Client(intents=discord.Intents.all())


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')
    
    global general
    general = client.get_channel(1039230555797061704)
    
    global botChannel
    botChannel = client.get_channel(1041048235378225192)
    
    global simon
    simon = await client.fetch_user(316949626697678858)

    global paul
    paul = await client.fetch_user(317684843393581056)
    
    await botChannel.send(":green_circle: \n ca marche")


@client.event
async def on_message(message):
        
    if message.author == simon:
        message.channel.send("Ta gueule Simon")

client.run(TOKEN)

# await client.close()