%JSON_ENCODE encodes a MATLAB object into a JSON string.
%   Structures are converted to JSON objects and arrays are converted
%   to JSON arrays. Inf values are converted to the string "Inf". NaN values
%   are converted to 'null'.
%
%   Example:
%     s = struct();
%     s.patient.name = 'John Doe';
%     s.patient.billing = 127.00;
%     s.patient.test = [79, 75, 73; 180, 178, 177.5; 220, 210, 205];
%     s.patient(2).name = 'Ann Lane';
%     s.patient(2).billing = 28.50;
%     s.patient(2).test = [68, 70, 68; 118, 118, 119; 172, 170, 169];
%     s.patient(3).name = 'New Name';
%     disp(json_encode(s));
%
%   Note:
%     This function implements a superset of JSON as specified in the original
%     RFC 4627 - it will also encode scalar types and NULL. RFC 4627 only
%     supports these values when they are nested inside an array or an object.
%     Although this superset is consistent with the expanded definition of
%     "JSON text" in the newer RFC 7159 (which aims to supersede RFC 4627),
%     this may cause interoperability issues with older JSON parsers that
%     adhere strictly to RFC 4627 when encoding a single scalar value.
%     See http://www.rfc-editor.org/rfc/rfc7159.txt for more information.

%  Created by Léa Strobino.
%  Copyright 2018. All rights reserved.
