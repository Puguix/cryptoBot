import discord
import time

TOKEN = "MTAzOTIyOTc5Njk2Nzc4ODY4NA.GEBYR2.bQWmWep-AqaeP8YKBZoWNzVp8AlXZf6z089BRk"

messages = "Mon bot vient de dire quelque chose..."

client = discord.Client(intents=discord.Intents.default())
@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

    channel = client.get_channel(1041048235378225192)
    await channel.send(messages)

    user = await client.fetch_user(317684843393581056)
    await user.send(messages)

    await client.close()
    time.sleep(1)


client.run(TOKEN)