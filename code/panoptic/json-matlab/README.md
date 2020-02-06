# matlab-json

**json_decode** parses a JSON string and returns a MATLAB object.
JSON objects are converted to structures and JSON arrays are converted to
vectors (all elements of the same type) or cell arrays (different types).
'null' values are converted to NaN.

**json_encode** encodes a MATLAB object into a JSON string.
Structures are converted to JSON objects and arrays are converted
to JSON arrays. Inf values are converted to the string "Inf". NaN values
are converted to 'null'.

### Note
This function implements a superset of JSON as specified in the original
RFC 4627 - it will also decode scalar types and NULL. RFC 4627 only
supports these values when they are nested inside an array or an object.
Although this superset is consistent with the expanded definition of
"JSON text" in the newer RFC 7159 (which aims to supersede RFC 4627),
this may cause interoperability issues with older JSON parsers that
adhere strictly to RFC 4627 when encoding a single scalar value.
See http://www.rfc-editor.org/rfc/rfc7159.txt for more information.

## Compilation
```
mex json_decode.c jsmn.c
mex json_encode.c
```

## Example
```
url = 'https://aviationweather.gov/gis/scripts/MetarJSON.php?bbox=6.11,46.23,6.12,46.24';
metar = json_decode(urlread(url));
disp(metar.features.properties);

s = struct();
s.patient.name = 'John Doe';
s.patient.billing = 127.00;
s.patient.test = [79, 75, 73; 180, 178, 177.5; 220, 210, 205];
s.patient(2).name = 'Ann Lane';
s.patient(2).billing = 28.50;
s.patient(2).test = [68, 70, 68; 118, 118, 119; 172, 170, 169];
s.patient(3).name = 'New Name';
disp(json_encode(s));
```
