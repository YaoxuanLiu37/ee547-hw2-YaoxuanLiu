#!/usr/bin/env python3
import boto3, json, sys, os, argparse
from datetime import datetime, UTC
def ts(): return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def warn(msg): print(f"[WARNING] {msg}", file=sys.stderr)
def err(msg):  print(f"[ERROR] {msg}",  file=sys.stderr)

def retry_once(fn, *a, **k):
    try: return fn(*a, **k)
    except: return fn(*a, **k)

def validate_region_or_exit(session, region):
    if not region: return session.region_name or "us-east-1"
    try: avail = set(session.get_available_regions("ec2"))
    except: avail = set()
    if region not in avail:
        err(f"Invalid region: {region}")
        sys.exit(1)
    return region

def get_caller_identity(sts):
    try:
        r = retry_once(sts.get_caller_identity)
        return r.get("Account"), r.get("Arn")
    except:
        err("Authentication failed. Ensure AWS credentials are configured.")
        sys.exit(1)

def collect_iam_users(iam):
    users = []
    try:
        paginator = iam.get_paginator("list_users")
        for page in retry_once(paginator.paginate):
            for u in page.get("Users", []) or []:
                user = {
                    "username": u.get("UserName",""),
                    "user_id": u.get("UserId",""),
                    "arn": u.get("Arn",""),
                    "create_date": u.get("CreateDate").strftime("%Y-%m-%dT%H:%M:%SZ") if u.get("CreateDate") else None,
                    "last_activity": None,
                    "attached_policies": []
                }
                try:
                    gu = retry_once(iam.get_user, UserName=user["username"])
                    pl = gu.get("User", {}).get("PasswordLastUsed")
                    if pl: user["last_activity"] = pl.strftime("%Y-%m-%dT%H:%M:%SZ")
                except: pass
                try:
                    pols = retry_once(iam.list_attached_user_policies, UserName=user["username"]).get("AttachedPolicies", [])
                    for p in pols:
                        user["attached_policies"].append({
                            "policy_name": p.get("PolicyName",""),
                            "policy_arn": p.get("PolicyArn","")
                        })
                except: pass
                users.append(user)
    except:
        warn("Access denied for IAM operations - skipping user enumeration")
        return []
    if not users: warn("No IAM users found")
    return users

def collect_ec2_instances(ec2, region):
    insts = []
    try: res = retry_once(ec2.describe_instances)
    except: 
        warn(f"EC2 describeInstances failed in {region}")
        return []
    try:
        for r in res.get("Reservations", []) or []:
            for i in r.get("Instances", []) or []:
                item = {
                    "instance_id": i.get("InstanceId",""),
                    "instance_type": i.get("InstanceType"),
                    "state": (i.get("State") or {}).get("Name",""),
                    "public_ip": i.get("PublicIpAddress"),
                    "private_ip": i.get("PrivateIpAddress"),
                    "availability_zone": (i.get("Placement") or {}).get("AvailabilityZone"),
                    "launch_time": i.get("LaunchTime").strftime("%Y-%m-%dT%H:%M:%SZ") if i.get("LaunchTime") else None,
                    "ami_id": i.get("ImageId"),
                    "ami_name": None,
                    "security_groups": [sg.get("GroupId") for sg in (i.get("SecurityGroups") or [])],
                    "tags": {t.get("Key"): t.get("Value") for t in (i.get("Tags") or []) if t.get("Key") and t.get("Value")}
                }
                if item["ami_id"]:
                    try:
                        imgs = retry_once(ec2.describe_images, ImageIds=[item["ami_id"]]).get("Images", [])
                        if imgs: item["ami_name"] = imgs[0].get("Name")
                    except: pass
                insts.append(item)
    except:
        warn(f"EC2 parsing failed in {region}")
        return []
    if not insts: warn(f"No EC2 instances found in {region}")
    return insts

def collect_s3_buckets(s3):
    buckets = []
    try: lb = retry_once(s3.list_buckets)
    except:
        warn("Access denied for S3 operations - skipping bucket enumeration")
        return []
    for b in lb.get("Buckets", []) or []:
        name = b.get("Name")
        create = b.get("CreationDate")
        rec = {
            "bucket_name": name,
            "creation_date": create.strftime("%Y-%m-%dT%H:%M:%SZ") if create else None,
            "region": None,
            "object_count": 0,
            "size_bytes": 0
        }
        try:
            loc = retry_once(s3.get_bucket_location, Bucket=name)
            lc = loc.get("LocationConstraint")
            rec["region"] = "us-east-1" if (lc in (None, "")) else lc
        except: pass
        try:
            paginator = s3.get_paginator("list_objects_v2")
            total_n, total_sz = 0, 0
            for page in retry_once(paginator.paginate, Bucket=name):
                for obj in page.get("Contents", []) or []:
                    total_n += 1
                    total_sz += int(obj.get("Size", 0))
            rec["object_count"] = total_n
            rec["size_bytes"] = total_sz
        except:
            err(f"Failed to access S3 bucket '{name}': Access Denied")
        buckets.append(rec)
    if not buckets: warn("No S3 buckets found")
    return buckets

def sg_rule_to_text(ip):
    proto = ip.get("IpProtocol", "all")
    if proto == "-1": proto = "all"
    def pr(a,b): return "all" if (a is None or b is None) else f"{a}-{b}"
    srcs=[]
    for r in ip.get("IpRanges", []) or []: srcs.append(r.get("CidrIp"))
    for r in ip.get("Ipv6Ranges", []) or []: srcs.append(r.get("CidrIpv6"))
    for r in ip.get("UserIdGroupPairs", []) or []: srcs.append(r.get("GroupId"))
    return proto, pr(ip.get("FromPort"), ip.get("ToPort")), ", ".join([s for s in srcs if s])

def collect_security_groups(ec2, region):
    out=[]
    try: resp = retry_once(ec2.describe_security_groups)
    except:
        warn(f"EC2 describeSecurityGroups failed in {region}")
        return []
    for g in resp.get("SecurityGroups", []) or []:
        inbound=[]
        for ip in g.get("IpPermissions", []) or []:
            proto, prange, src = sg_rule_to_text(ip)
            inbound.append({"protocol": proto, "port_range": prange, "source": src or "-"})
        outbound=[]
        for ip in g.get("IpPermissionsEgress", []) or []:
            proto, prange, dst = sg_rule_to_text(ip)
            outbound.append({"protocol": proto, "port_range": prange, "destination": dst or "-"})
        out.append({
            "group_id": g.get("GroupId",""),
            "group_name": g.get("GroupName"),
            "description": g.get("Description"),
            "vpc_id": g.get("VpcId"),
            "inbound_rules": inbound,
            "outbound_rules": outbound
        })
    if not out: warn(f"No security groups found in {region}")
    return out

def to_table(doc):
    acc = doc["account_info"]; res = doc["resources"]
    lines=[]
    lines.append(f"AWS Account: {acc.get('account_id','-')} ({acc.get('region','-')})")
    lines.append(f"Scan Time: {acc.get('scan_timestamp','-').replace('T',' ').replace('Z',' UTC')}")
    lines.append("")
    users = res.get("iam_users", [])
    lines.append(f"IAM USERS ({len(users)} total)")
    lines.append(f"{'Username':<20} {'Create Date':<20} {'Last Activity':<20} {'Policies':>8}")
    for u in users:
        lines.append(
            f"{u.get('username',''):<20} "
            f"{str(u.get('create_date') or '-')[:10]:<20} "
            f"{str(u.get('last_activity') or '-')[:10]:<20} "
            f"{len(u.get('attached_policies', [])):>8}"
        )
    lines.append("")
    ec2 = res.get("ec2_instances", [])
    running = sum(1 for i in ec2 if i.get("state")=="running")
    lines.append(f"EC2 INSTANCES ({running} running, {len(ec2)-running} stopped)")
    lines.append(f"{'Instance ID':<20} {'Type':<12} {'State':<10} {'Public IP':<16} {'Launch Time':<16}")
    for i in ec2:
        lt = str(i.get('launch_time') or '-')[:16].replace('T', ' ')
        pub = str(i.get('public_ip') or '-')
        lines.append(
            f"{i.get('instance_id',''):<20} "
            f"{i.get('instance_type',''):<12} "
            f"{i.get('state',''):<10} "
            f"{pub:<16} "
            f"{lt:<16}"
        )
    lines.append("")
    s3 = res.get("s3_buckets", [])
    lines.append(f"S3 BUCKETS ({len(s3)} total)")
    lines.append(f"{'Bucket Name':<24} {'Region':<10} {'Created':<13} {'Objects':>10} {'Size (MB)':>10}")
    for b in s3:
        created = str(b.get('creation_date') or '-')[:10]
        size_mb = (b.get('size_bytes', 0) / 1024 / 1024)
        size_str = f"~{size_mb:.1f}" if size_mb > 0 else "0"
        lines.append(
            f"{b.get('bucket_name',''):<24} "
            f"{str(b.get('region') or '-'):<10} "
            f"{created:<13} "
            f"{int(b.get('object_count', 0)):>10} "
            f"{size_str:>10}"
        )
    lines.append("")
    sgs = res.get("security_groups", [])
    lines.append(f"SECURITY GROUPS ({len(sgs)} total)")
    lines.append(f"{'Group ID':<16} {'Name':<14} {'VPC ID':<14} {'Inbound Rules':>14}")
    for g in sgs:
        lines.append(
            f"{g.get('group_id',''):<16} "
            f"{g.get('group_name',''):<14} "
            f"{str(g.get('vpc_id') or '-'):<14} "
            f"{len(g.get('inbound_rules', [])):>14}"
        )
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--format", choices=["json","table"], default="json")
    args = ap.parse_args()
    session = boto3.Session()
    region = validate_region_or_exit(session, args.region)
    session = boto3.Session(region_name=region)
    account_id, user_arn = get_caller_identity(session.client("sts"))
    iam = session.client("iam")
    ec2 = session.client("ec2", region_name=region)
    s3 = session.client("s3")
    iam_users = collect_iam_users(iam)
    ec2_instances = collect_ec2_instances(ec2, region)
    s3_buckets = collect_s3_buckets(s3)
    sec_groups = collect_security_groups(ec2, region)
    doc = {
        "account_info": {
            "account_id": account_id,
            "user_arn": user_arn,
            "region": region,
            "scan_timestamp": ts()
        },
        "resources": {
            "iam_users": iam_users,
            "ec2_instances": ec2_instances,
            "s3_buckets": s3_buckets,
            "security_groups": sec_groups
        },
        "summary": {
            "total_users": len(iam_users),
            "running_instances": sum(1 for i in ec2_instances if i.get("state")=="running"),
            "total_buckets": len(s3_buckets),
            "security_groups": len(sec_groups)
        }
    }
    if args.format == "json":
        payload = json.dumps(doc, ensure_ascii=False, indent=2)
    else:
        payload = to_table(doc)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f: f.write(payload)
    else:
        print(payload)

if __name__ == "__main__":
    main()
