import asyncio
import sys
from playwright.async_api import async_playwright


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome")
        pg = await b.new_page(viewport={"width": 1360, "height": 1520}, device_scale_factor=2)
        await pg.goto("file:///path/to/project/an2/fig1design/project/Main.dc.html")
        await pg.wait_for_timeout(1500)
        await pg.screenshot(path="fig/NH3_Fig1.png", full_page=False)
        await pg.pdf(
            path="fig/NH3_Fig1.pdf", width="1360px", height="1520px", print_background=True
        )
        await b.close()


asyncio.run(main())
