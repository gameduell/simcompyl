#!/bin/sed -nf
# this files converts a coverage.xml report into a basic lcov report
/^\s*<source>/{
  s/^\s*<source>\([^<]*\)<\/source>/\1\//
  h
}
1{
  s/.*/TN:Test #py/
  p
}
/^\s*<class\s/{
  G
  s/.*filename="\([^"]*\)".*\n\(.*\)/SF:\2\1/
  aFN:1,py
  aFNDA:1,py
  aFNF:1
  aFNH:1
  p
}
/^\s*<line\s/{
  s/.*hits="\([^"]*\)".*number="\([^"]*\)".*/DA:\2,\1/
  p
}
/^\s*<\/class>/{
  s/.*/LF:1\nLH:1\nend_of_record/
  p
}
