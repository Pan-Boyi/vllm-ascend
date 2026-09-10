import json,subprocess,sys
def ls(p):
    out=subprocess.run(["curl","-sSL","--max-time","30",f"https://api.gitcode.com/api/v5/repos/cann/cann-recipes-infer/contents/{p}"],capture_output=True,text=True).stdout
    try: return json.loads(out)
    except Exception: return []
def walk(p,depth=0,maxd=6):
    for e in ls(p):
        print(e['type'],e['path'],e.get('download_url','') if e['type']=='file' else '')
        if e['type']=='dir' and depth<maxd: walk(e['path'],depth+1,maxd)
walk(sys.argv[1])
